import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model

from models.ced.audiotransformer import AudioTransformer


from models.modeling_llama import LlamaForCausalLM

from models.Qformer import BertConfig, BertLMHeadModel
from models.utils import StoppingCriteriaSub, log_optimal_transport

from loss.SmoothLabelCE import SmoothLabelCrossEntropyLoss
import numpy as np

################################################################################
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
################################################################################

class LAVCAP(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2, freeze_speech_QFormer=False):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # added by khrho
        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if freeze_speech_QFormer:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer.eval()
            query_tokens.requires_grad = False
            logging.info("freeze Speech QFormer")

        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        encoder_path="",
        freeze_enc=True,

        llama_decoder_path="",

        n_class=527,
        embed_dim=768,
        vis_embed_dim=768,
        melbins=64,
        target_length=1024,

        enc_lora=True,
        enc_lora_rank=8,
        enc_lora_alpha=32,
        enc_lora_dropout=0.1,

        dec_lora=True,
        dec_lora_rank=8,
        dec_lora_alpha=32,
        dec_lora_dropout=0.1,

        #################################
        fusion_type = "chan_cat_bfQF",
        spec_f_mean = False,
        av_embed_norm = False,
        ot_loss_weight=0.1,
        num_sinkhorn_iter=50,
        sinkhorn_epsilon=0.1,
        prompt_ratio=0,
        use_exp_Q=False,
        #################################
        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,

        speech_llama_proj_model = "",
        freeze_speech_llama_proj = False,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",

        loss_type=None,
        padding_idx=0,
        
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        
        super().__init__()    

        self.embed_dim = embed_dim
        self.vis_embed_dim=vis_embed_dim

        self.melbins = melbins
        self.target_length = target_length

        self.enc_lora = enc_lora
        self.dec_lora = dec_lora

        ##########################################
        self.fusion_type = fusion_type
        self.spec_f_mean = spec_f_mean
        self.av_embed_norm = av_embed_norm
        self.ot_loss_weight = ot_loss_weight
        self.num_sinkhorn_iter = num_sinkhorn_iter
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.prompt_ratio = prompt_ratio
        self.use_exp_Q = use_exp_Q
        if self.ot_loss_weight > 0:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # learnable temperature $\tau$ initialized as 0.07
            self.celoss = nn.CrossEntropyLoss()
        ##########################################

        self.use_speech_Qformer = use_speech_Qformer  
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride  

        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.loss_type = loss_type
        self.padding_idx = padding_idx
        

        self.low_resource = low_resource
        
        self.img_info = {"num_has_image": 0, "num_no_image": 0}

        if self.av_embed_norm:
            self.ln_aud = nn.LayerNorm(self.embed_dim)
            self.ln_vis = nn.LayerNorm(self.embed_dim)

        logging.info("Loading CED Encoder")

        self.audio_encoder = AudioTransformer(output_dim=n_class, n_mels=self.melbins, target_length=self.target_length,
                                              patch_size=16, embed_dim=self.embed_dim, depth=12, num_heads=12, pooling="mean",
                                              time_patch_out=None, freq_patch_out=None)

        if encoder_path is not None:
            self.audio_encoder.load_state_dict(encoder_path)

        if freeze_enc:
            logging.info("Freeze CED Encoder\n")
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
        else: 
            logging.info("Fine-tune CED Encoder\n")

        if self.enc_lora:
            logging.info("Encoder LoRA Setting...")
            self.enc_peft_config = LoraConfig(
                # task_type=TaskType.CAUSAL_LM, # Decoder
                # use_dora = True,
                target_modules=["q_proj", "v_proj"],
                inference_mode=False, 
                r=enc_lora_rank, 
                lora_alpha=enc_lora_alpha, 
                lora_dropout=enc_lora_dropout,
            )
            self.audio_encoder = get_peft_model(self.audio_encoder, self.enc_peft_config)
            self.audio_encoder.print_trainable_parameters()
            logging.info("Encoder LoRA Training\n")

        logging.info("Loading LLaMA Tokenizer")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_decoder_path, use_fast=False, legacy=False)
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"

        logging.info("Loading LLaMA Model")
        if self.low_resource:
            self.llama_decoder = LlamaForCausalLM.from_pretrained(
                llama_decoder_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_decoder = LlamaForCausalLM.from_pretrained(
                llama_decoder_path,
                torch_dtype=torch.float16,
            )
        self.llama_decoder.resize_token_embeddings(len(self.llama_tokenizer))

        logging.info("Freezing LLaMA Model\n")
        for name, param in self.llama_decoder.named_parameters():
            param.requires_grad = False
            
        if self.dec_lora:
            logging.info("Decoder LoRA Setting...")
            self.dec_peft_config = LoraConfig(
                # use_dora = True,
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=dec_lora_rank, 
                lora_alpha=dec_lora_alpha, 
                lora_dropout=dec_lora_dropout,
            )
            self.llama_decoder = get_peft_model(self.llama_decoder, self.dec_peft_config)
            self.llama_decoder.print_trainable_parameters()
            logging.info("Decoder LoRA Training\n")

        if self.use_speech_Qformer:
            if "chancat_bfQF" in self.fusion_type:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=2*self.embed_dim,
                freeze_speech_QFormer=freeze_speech_QFormer)
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.embed_dim,
                freeze_speech_QFormer=freeze_speech_QFormer)

            logging.info("Loading speech LLaMA proj")
            if "chan" in self.fusion_type: # chancat_bfQF, chancat_afQF, ot_chancat_bfQF, ot_chancat_afQF
                self.speech_llama_proj = nn.Linear(
                    2*self.embed_dim, self.llama_decoder.config.hidden_size
                )
            elif "temp" in self.fusion_type: # tempcat_bfQF, tempcat_afQF, ot_tempcat_bfQF, ot_tempcat_afQF
                self.speech_llama_proj = nn.Linear(
                    self.embed_dim, self.llama_decoder.config.hidden_size
                )
            if speech_llama_proj_model:
                logging.info("Loading speech LLaMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight["model"], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLaMA proj")
            
        else:
            if "chan" in self.fusion_type: # chancat_bfQF, chancat_afQF, ot_chancat_bfQF, ot_chancat_afQF
                self.speech_llama_proj = nn.Linear(
                    2*self.embed_dim, self.llama_decoder.config.hidden_size
                )
            elif "temp" in self.fusion_type: # tempcat_bfQF, tempcat_afQF, ot_tempcat_bfQF, ot_tempcat_afQF
                self.speech_llama_proj = nn.Linear(
                    self.embed_dim, self.llama_decoder.config.hidden_size
                )
            elif "crossattn" in self.fusion_type:
                self.cross_attn = nn.MultiheadAttention(embed_dim, embed_dim//64, batch_first=True)
                self.speech_llama_proj = nn.Linear(
                    self.embed_dim, self.llama_decoder.config.hidden_size
                )
        
        if self.fusion_type == "tempcat_afQF":
            self.visual_llama_proj= nn.Linear(vis_embed_dim, self.llama_decoder.config.hidden_size) # 768 -> 4096

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

        logging.info(f"Defining Loss Term")
        if loss_type is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif loss_type == "smooth":
            self.criterion = SmoothLabelCrossEntropyLoss(ignore_index=self.padding_idx)
        else:
            raise NotImplementedError(loss_type)

    def _encode_auditory_feature(self, spec_embeds, vis_embeds):
        # spec_embeds : batch x num_patchs(melbins//16 x target_length//16) x embed_dim
        # clip_feat : batch x 20(frames) x embed_dim

        with self.maybe_autocast():
            ###########  edited by lhk  ###########
            # Normalize the audio and visual features
            if self.av_embed_norm:
                spec_embeds = self.ln_aud(spec_embeds)
                vis_embeds = self.ln_vis(vis_embeds)

            # Mean pooling along the frequency axis
            if self.spec_f_mean:
                spec_embeds = spec_embeds.view(spec_embeds.size(0),self.melbins//16,self.target_length//16,spec_embeds.size(-1))
                spec_embeds = torch.mean(spec_embeds,dim=1) # B x TA x C

            B, TA, C = spec_embeds.shape # batch x temporal(audio) x embed_dim
            B, TV, C = vis_embeds.shape # batch x temporal(visual) x embed_dim
            ot_loss = 0

            # Audio-visual fusion
            if "ot" in self.fusion_type:
                # vis_embeds shape : B x TV x 768
                # audio_embeds shape : B x TA x 768

                if self.use_speech_Qformer:
                    kernel = round(64 * self.second_per_window / 10.0) # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
                    stride = round(64 * self.second_stride / 10.0)
                    kernel, stride = (1, kernel), (1, stride)
                    spec_embeds_tr = spec_embeds.transpose(1, 2).unsqueeze(2)
                    spec_embeds_overlap = F.unfold(spec_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = spec_embeds_overlap.shape
                    spec_embeds_overlap = spec_embeds_overlap.view(B, -1, kernel[1], L)
                    spec_embeds_overlap = torch.permute(spec_embeds_overlap, [0, 3, 2, 1])
                    spec_embeds = spec_embeds_overlap.reshape(-1, kernel[1], C)
                    av_atts = torch.ones(spec_embeds.size()[:-1], dtype=torch.long, device=spec_embeds.device)

                    query_tokens = self.speech_query_tokens.expand(spec_embeds.shape[0], -1, -1)
                    query_output = self.speech_Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=spec_embeds,
                        encoder_attention_mask=av_atts,
                        return_dict=True,
                    )
                    spec_embeds = query_output.last_hidden_state
                    spec_embeds = spec_embeds.view(B, -1, spec_embeds.size(2)).contiguous()[:,:-1,:]
                    B, TA, C = spec_embeds.shape # batch x temporal(audio) x embed_dim

                # normalize the audio and visual features
                vis_embeds = vis_embeds / vis_embeds.norm(dim=-1, keepdim=True)     # B x TV x 768
                spec_embeds = spec_embeds / spec_embeds.norm(dim=-1, keepdim=True)  # B x TA x 768

                # sinkhorn algorithm
                va_sim = torch.einsum("bad,bvd->bav", vis_embeds, spec_embeds)  # B x TV x TA
                Q_v = torch.zeros_like(va_sim)    # B x TV x TA
                for i in range(B):
                    Q_v[i] = log_optimal_transport(va_sim[i] / self.sinkhorn_epsilon, self.num_sinkhorn_iter)  # B x TV x TA
                Q_a = Q_v.transpose(1, 2)  # B x TA x TV

                # calculate the loss
                with torch.no_grad():
                    self.logit_scale.clamp_(0, 4.6052)  # $\tau$ between 0.01, 1

                if self.use_exp_Q:
                    logits_v = self.logit_scale * torch.einsum("bva,bua->bvu", Q_v.exp(), va_sim) # B x TV x TV
                    logits_a = self.logit_scale * torch.einsum("bav,buv->bau", Q_a.exp(), va_sim.transpose(1,2)) # B x TA x TA
                else:
                    logits_v = self.logit_scale * torch.einsum("bva,bua->bvu", Q_v, va_sim) # B x TV x TV
                    logits_a = self.logit_scale * torch.einsum("bav,buv->bau", Q_a, va_sim.transpose(1,2)) # B x TA x TA

                targets_v = torch.arange(logits_v.size(1), dtype=torch.long).unsqueeze(0).expand(logits_v.size(0), -1).to(va_sim.device)  # B x TV
                targets_a = torch.arange(logits_a.size(1), dtype=torch.long).unsqueeze(0).expand(logits_a.size(0), -1).to(va_sim.device)  # B x TA

                ot_loss_v = self.ot_loss_weight * self.celoss(logits_v, targets_v)
                ot_loss_a = self.ot_loss_weight * self.celoss(logits_a, targets_a)
                ot_loss = ot_loss_v + ot_loss_a

                #channel cat
                if "ot_chancat" in self.fusion_type:
                    if self.use_speech_Qformer:
                        av_embeds = torch.cat((spec_embeds, vis_embeds), dim=-1)
                    else:   # padding
                        exp_vis_embeds = []
                        gap, rem = TA//TV, TA%TV

                        for i in range(len(vis_embeds[0])):
                            vis_elem = vis_embeds[:,i].view(B,-1,C)
                            vis_elem = F.pad(vis_elem,(0,0,0,gap-1))
                            exp_vis_embeds.append(vis_elem)
                        exp_vis_embeds = torch.cat(exp_vis_embeds,dim=1)
                        exp_vis_embeds = F.pad(exp_vis_embeds,(0,0,0,rem))

                        av_embeds = torch.cat((spec_embeds, exp_vis_embeds), dim=-1)

                elif "ot_tempcat" in self.fusion_type:
                    av_embeds = torch.cat([spec_embeds, vis_embeds], dim=1)  # B x (TV+TA) x 768

                    if "QAV_add" in self.fusion_type:
                        QV = torch.einsum("bva,bvd->bad",torch.exp(Q_v), vis_embeds) # B x TA x 768
                        QA = torch.einsum("bav,bad->bvd",torch.exp(Q_a), spec_embeds) # B x TV x 768
                        
                        QVA_embeds = torch.cat([QV, QA], dim=1)
                        av_embeds = av_embeds + QVA_embeds    # B x (TV+TA) x 768

                elif "crossattn" in self.fusion_type:
                    av_embeds, _ = self.cross_attn(spec_embeds, vis_embeds, vis_embeds)
                    av_embeds = av_embeds + spec_embeds

                # project av_embeds to llama latent space
                av_embeds = self.speech_llama_proj(av_embeds)

            # Not use Optimal Transport
            else:
                if self.fusion_type == "tempcat_bfQF":
                    av_embeds = torch.cat((spec_embeds,vis_embeds), dim=1)

                elif self.fusion_type == "chancat_bfQF":
                    exp_vis_embeds = []
                    gap, rem = TA//TV, TA%TV

                    for i in range(len(vis_embeds[0])):
                        vis_elem = vis_embeds[:,i].view(B,-1,C)
                        vis_elem = F.pad(vis_elem,(0,0,0,gap-1))
                        exp_vis_embeds.append(vis_elem)
                    exp_vis_embeds = torch.cat(exp_vis_embeds,dim=1)
                    exp_vis_embeds = F.pad(exp_vis_embeds,(0,0,0,rem))

                    av_embeds = torch.cat((spec_embeds,exp_vis_embeds), dim=2)

                elif self.fusion_type == "aud_embed_only":
                    av_embeds = spec_embeds

                elif self.fusion_type == "vis_embed_only":
                    av_embeds = vis_embeds
                
                elif self.fusion_type == "tempcat_afQF":
                    av_embeds = spec_embeds

                else:
                    # raise NotImplementedError
                    av_embeds = spec_embeds

                av_atts = torch.ones(av_embeds.size()[:-1], dtype=torch.long).to(av_embeds.device)

                if self.use_speech_Qformer:
                    if self.window_level_Qformer:
                        B, T, C = av_embeds.shape  # batch, tokens(temporal), channel
                        kernel = round(1500 * self.second_per_window / 30.0) # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
                        stride = round(1500 * self.second_stride / 30.0)
                        kernel, stride = (1, kernel), (1, stride)
                        av_embeds_tr = av_embeds.transpose(1, 2).unsqueeze(2)
                        av_embeds_overlap = F.unfold(av_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                        _, _, L = av_embeds_overlap.shape
                        av_embeds_overlap = av_embeds_overlap.view(B, -1, kernel[1], L)
                        av_embeds_overlap = torch.permute(av_embeds_overlap, [0, 3, 2, 1])
                        av_embeds = av_embeds_overlap.reshape(-1, kernel[1], C)
                        av_atts = torch.ones(av_embeds.size()[:-1], dtype=torch.long, device=av_embeds.device)

                    query_tokens = self.speech_query_tokens.expand(av_embeds.shape[0], -1, -1)
                    query_output = self.speech_Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=av_embeds,
                        encoder_attention_mask=av_atts,
                        return_dict=True,
                    )
                    av_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                    if self.window_level_Qformer:
                        av_embeds = av_embeds.view(B, -1, av_embeds.size(2)).contiguous()

                    if self.fusion_type == 'temp_cat_afQF':
                        vis_embeds = self.visual_llama_proj(vis_embeds)
                        av_embeds = torch.cat([av_embeds, vis_embeds], dim=1)

                else:   # no Qformer used
                    if "temp" in self.fusion_type:
                        av_embeds = torch.cat([av_embeds, vis_embeds], dim=1)  # B x (TV+TA) x 768
                    av_embeds = self.speech_llama_proj(av_embeds)

            av_atts = torch.ones(av_embeds.size()[:-1], dtype=torch.long).to(av_embeds.device)

        return av_embeds, av_atts, ot_loss

    def encode_feature(self, spec, clip_feat):
        # edited by khrho
        with self.maybe_autocast():
            spec_embeds = self.audio_encoder(spec)
            av_embeds, av_atts, ot_loss = self._encode_auditory_feature(spec_embeds, clip_feat)

        return av_embeds, av_atts, ot_loss

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        B = len(embeds)
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_decoder.model.embed_tokens(p_before_tokens.input_ids) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_decoder.model.embed_tokens(p_after_tokens.input_ids) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(p_after_tokens.input_ids)
  
                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_decoder.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_decoder.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts
    
    def forward(self, batch, verbose=False):
        spec, clip_feat = batch["spec"], batch["clip_feat"]
        av_embeds, av_atts, ot_loss = self.encode_feature(spec, clip_feat)

        batch_size = len(av_embeds)

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = []
                for i in range(batch_size):
                    prompt.append(random.choice(self.prompt_dict["audiocaption"]))                        
            else:
                prompt = random.choice(self.prompt_dict["audiocaption"])

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            av_embeds, av_atts = self.prompt_wrap(av_embeds, av_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in batch["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(av_embeds.device)
        to_regress_embeds = self.llama_decoder.model.embed_tokens(to_regress_tokens.input_ids) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([av_atts.shape[0], av_atts.shape[1] + 1],dtype=torch.long
            ).to(av_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_decoder.model.embed_tokens(bos) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(bos)
        atts_bos = av_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, av_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, av_atts, to_regress_tokens.attention_mask], dim=1)
        
        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss + ot_loss   # edited by khrho

        if verbose:
            nvocab = self.llama_decoder.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, batch, generate_cfg, prompts=None, search_param=None):
        spec, clip_feat = batch["spec"], batch["clip_feat"]
        av_embeds, av_atts, _ = self.encode_feature(spec, clip_feat)
        
        batch_size = len(av_embeds)

        if prompts is not None:
            av_embeds, av_atts = self.prompt_wrap(av_embeds, av_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=av_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_decoder.model.embed_tokens(bos) if not self.dec_lora else self.llama_decoder.model.model.embed_tokens(bos)
        atts_bos = av_atts[:, :1]

        embeds = torch.cat([bos_embeds, av_embeds], dim=1)
        attns = torch.cat([atts_bos, av_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_decoder.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        # predicted_caption = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        pad_num = generate_cfg.get("max_new_tokens", 200) - len(outputs[0])
        outputs = F.pad(outputs, (0,pad_num), 'constant', 0)

        return {"torch_id": batch["torch_id"],"predictions": outputs}
        # return {"torch_id": batch["torch_id"],"predictions": predicted_caption,"logprobs": logprobs}

    @classmethod
    def from_config(cls, config):
        encoder_path = config.get("encoder_path","")
        freeze_enc = config.get("freeze_enc",True)
        llama_decoder_path = config.get("llama_decoder_path","")

        n_class = config.get("n_class", 527)
        embed_dim = config.get("embed_dim", 768)
        melbins = config.get("melbins", 128)
        target_length = config.get("target_length", 1024)

        enc_lora = config.get("enc_lora", True)
        enc_lora_rank = config.get("enc_lora_rank", 8)
        enc_lora_alpha = config.get("enc_lora_alpha", 32)
        enc_lora_dropout = config.get("enc_lora_dropout", 0.1)

        dec_lora = config.get("dec_lora", True)
        dec_lora_rank = config.get("dec_lora_rank", 8)
        dec_lora_alpha = config.get("dec_lora_alpha", 32)
        dec_lora_dropout = config.get("dec_lora_dropout", 0.1)

        # Fusion (Optimal Transport)
        fusion_type = config.get("fusion_type", "chan_cat_bfQF")
        spec_f_mean = config.get("spec_f_mean", False)
        av_embed_norm = config.get("av_embed_norm", False)
        ot_loss_weight = config.get("ot_loss_weight", 0.1)
        num_sinkhorn_iter = config.get("num_sinkhorn_iter", 50)
        sinkhorn_epsilon = config.get("sinkhorn_epsilon", 0.1)
        prompt_ratio = config.get("prompt_ratio", 0)
        use_exp_Q = config.get("use_exp_Q", False)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")

        loss_type = config.get("loss_type", None)
        padding_idx = config.get("padding_idx", 0)

        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            encoder_path=encoder_path,
            freeze_enc=freeze_enc,
            llama_decoder_path=llama_decoder_path,
            n_class=n_class,
            embed_dim=embed_dim,
            melbins=melbins,
            target_length=target_length,
            enc_lora=enc_lora,
            enc_lora_rank=enc_lora_rank,
            enc_lora_alpha=enc_lora_alpha,
            enc_lora_dropout=enc_lora_dropout,
            dec_lora=dec_lora,
            dec_lora_rank=dec_lora_rank,
            dec_lora_alpha=dec_lora_alpha,
            dec_lora_dropout=dec_lora_dropout,
            #######################################
            fusion_type=fusion_type,
            spec_f_mean=spec_f_mean,
            av_embed_norm=av_embed_norm,
            ot_loss_weight=ot_loss_weight,
            num_sinkhorn_iter=num_sinkhorn_iter,
            sinkhorn_epsilon=sinkhorn_epsilon,
            prompt_ratio=prompt_ratio,
            use_exp_Q=use_exp_Q,
            #########################################
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            loss_type=loss_type,
            padding_idx=padding_idx,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load MMCAPs ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        return model
