# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset.AudioCaps import AudioCaps
from runner import Runner
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main_worker(gpu, ngpus_per_node, cfg):
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    cfg.config.run.gpu = gpu
    cfg.config.run.ngpus_per_node = ngpus_per_node
    
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    cfg.pretty_print()

    # build model
    model = load_model(model_config)

    # build datasets
    datasets = {
        "train": AudioCaps(data_config, data_config.train_data_path),
        "valid": AudioCaps(data_config, data_config.valid_data_path),
    }

    # build runner
    runner = Runner(cfg, model, datasets, job_id)

    # train
    runner.train()

def main():
    # load config
    cfg = Config(parse_args())

    n_gpus = torch.cuda.device_count()

    if cfg.config.run.use_distributed:
        mp.spawn(main_worker, args=(n_gpus, cfg), nprocs=n_gpus)
    else:
        main_worker(0, None, cfg)

if __name__ == "__main__":
    main()