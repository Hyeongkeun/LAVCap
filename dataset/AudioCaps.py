import os
import csv
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor

##
import torchaudio
from transformers import BertTokenizer
import torchvision.transforms as T
import PIL
from PIL import Image
import random
from torch.utils.data.dataloader import default_collate

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


class AudioCaps(Dataset):
    def __init__(self, cfg, data_path):
        super().__init__()

        self.config = cfg

        self.train_mode = os.path.basename(data_path)

        self.audio_path = os.path.join(data_path,"waveforms")
        self.video_path = os.path.join(data_path,"frames")
        self.clip_path = os.path.join(os.path.dirname(data_path), "visual_feature", self.train_mode)
        self.json_path = os.path.join(os.path.dirname(data_path),self.train_mode+".json")
        self.label_csv = self.config.get("label_path")

        self.annotation = json.load(open(self.json_path, "r"))

        self.data = self.pro_data(self.annotation)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_text_len = 40

        self.target_length = self.config.get("target_length",1024)
        self.melbins = self.config.get("melbins",128)
        self.freqm = self.config.get("freqm",0) if self.train_mode == "train" else 0
        self.timem = self.config.get("timem",0) if self.train_mode == "train" else 0

        self.dataset_mean = self.config.get("dataset_mean")
        self.dataset_std = self.config.get("dataset_std")

        self.skip_norm = self.config.get("skip_norm",False)

        self.noise = self.config.get("noise", False) if self.train_mode == "train" else False

        self.index_dict = make_index_dict(self.label_csv)
        self.label_num = len(self.index_dict)

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [os.path.join(self.audio_path, data_json[i]["audio_path"]), 
                            data_json[i]["labels"], 
                            data_json[i]["video_path"],
                            data_json[i]["caption"],
                            data_json[i]["torch_id"]]
        data_np = np.array(data_json, dtype=str)
        return data_np

    def get_fbank(self, fbank):
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.dataset_mean) / (self.dataset_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        return fbank

    def get_text(self, caption):
        target_encoding = self.tokenizer(caption, padding='do_not_pad',add_special_tokens=False,
                                         truncation=True, max_length=self.max_text_len)
        need_predict = [1] * len(target_encoding['input_ids'])
        payload = target_encoding['input_ids']

        if len(payload) > self.max_text_len:
            payload = payload[-(self.max_text_len - 2):]
            need_predict = need_predict[-(self.max_text_len - 2):]

        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]

        return input_ids, need_predict

    def __getitem__(self, index):
        data_ind = self.data[index]
        datum = {}
        datum['audio_path'] = data_ind[0]
        datum['labels'] = data_ind[1]
        datum['video_path'] = data_ind[2]
        datum['caption'] = data_ind[3]
        datum['torch_id'] = int(data_ind[4])
        datum['clip_feat'] = os.path.join(self.clip_path, os.path.basename(datum['audio_path']).replace('.wav', '.npy'))

        label_indices = np.zeros(self.label_num)

        audio, sr = torchaudio.load(datum["audio_path"])
        audio = audio - audio.mean()

        fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        spec_pad = self.target_length - fbank.shape[0]

        # cut and pad
        if spec_pad > 0:
            m = torch.nn.ZeroPad2d((0,0,0,spec_pad))
            fbank = m(fbank)
        elif spec_pad < 0:
            fbank = fbank[0:self.target_length, :]

        fbank = self.get_fbank(fbank)
      

        ### CLIP feature (added by khrho)
        clip_feat = np.load(datum['clip_feat'])

        input_ids, need_predict = self.get_text(datum['caption'])

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        batch={'spec': fbank,
               'clip_feat': clip_feat,
               'text': datum['caption'],
               # 'caption_tokens': torch.tensor(input_ids), 
               'need_predict': torch.tensor(need_predict),
               'torch_id': torch.tensor(datum['torch_id'])}
        return batch

    def __len__(self):
        return len(self.annotation)

    def collater(self, batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
        ele = batch[0]
        if isinstance(ele, dict):
            return {key: self.collater([d[key] for d in batch]) for key in ele}
        elif isinstance(ele, (tuple, list)):
            return [self.collater(x) for x in zip(*batch)]
        else:
            if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
                if not all(b.shape == batch[0].shape for b in batch[1:]):
                    assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                    shape = torch.tensor([b.shape for b in batch])
                    max_shape = tuple(shape.max(dim=0)[0].tolist())
                    batch2 = []
                    for b in batch:
                        if any(c < m for c, m in zip(b.shape, max_shape)):
                            b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                            if b.dim() == 1:
                                b2[:b.shape[0]] = b
                            elif b.dim() == 2:
                                b2[:b.shape[0], :b.shape[1]] = b
                            elif b.dim() == 3:
                                b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                            else:
                                raise NotImplementedError
                            b = b2
                        batch2.append(b)
                    batch = batch2
            return default_collate(batch)