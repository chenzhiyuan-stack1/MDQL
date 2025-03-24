# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
# import gym
import numpy as np
import os
import torch
import json

# import d4rl
from diffusion.utils import utils
from diffusion.utils.data_sampler import Data_Sampler
from diffusion.utils.logger import logger, setup_logger
# from torch.utils.tensorboard import SummaryWriter

# from v14_iql.py
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import json
import pdb
import onnxruntime as ort

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
    'v14':                           {'lr': 3e-5, 'eta': 0.8,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 10.0, 'top_k': 1},
}

# hyperparameters from v14_iql.py
TensorBatch = List[torch.Tensor]
evaluation_dataset_path = '/home/czy/Schaferct/ALLdatasets/emulate'
ENUM = 20  # every 5 evaluation set
# 拿出ENUM个测试集？
small_evaluation_datasets = []
policy_dir_names = os.listdir(evaluation_dataset_path)
for p_t in policy_dir_names[:ENUM]:
    policy_type_dir = os.path.join(evaluation_dataset_path, p_t)
    for e_f_name in os.listdir(policy_type_dir)[:ENUM]:
        e_f_path = os.path.join(policy_type_dir, e_f_name)
        small_evaluation_datasets.append(e_f_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
USE_WANDB = 1
b_in_Mb = 1e6

MAX_ACTION = 20  # Mbps
STATE_DIM = 150
ACTION_DIM = 1

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

NORMAL_VECTOR = np.array(
        [
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
)

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "v14"
    seed: int = 42  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = './checkpoints_diffusionql'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL 的参数，部分可以用作 Diffusion QL 的参数
    buffer_size: int = 6_538_000  # Replay buffer size
    # buffer_size: int = 8711389  # Replay buffer size
    batch_size: int = 512  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "BWEC-Schaferct"
    group: str = "Diffusion-QL"
    name: str = "Diffusion-QL"
    
    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
            
def get_args():            
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=42, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")

    ### SRPO ###
    # parser.add_argument("--env_name", default="rtc_srpo")  # OpenAI gym environment name
    parser.add_argument(
        "--dataset_path",
        # default="/home/min414/data3/mstrain-id-123.pickle",
        default='/home/czy/Schaferct/mstrain-id-123.pickle'
    )
    parser.add_argument(
        "--add_data_path",
        # default="/home/min414/data3/mstrain-id-345.pickle",
        default='/home/czy/Schaferct/mstrain-id-345.pickle'
    )
    parser.add_argument(
        "--eval_data_path",
        default="/home/czy/Schaferct/ALLdatasets/emulate",
    )
    parser.add_argument(
        "--ckpt_path",
        default="/home/min414/data1/data1/Schaferct/rtc_srpo/checkpoints",
    ) 
    parser.add_argument(
        "--critic_path",
        default="/home/czy/Schaferct/critic_ckpt120000.pth",
    )
    # Sets Gym, PyTorch and Numpy seeds eval_interval
    parser.add_argument("--eval_num", default=8, type=int)
    parser.add_argument("--max_steps", default=int(2e6), type=int)
    parser.add_argument("--eval_interval", default=int(1e5), type=int)
    # parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--save_model", default=1, type=int)
    parser.add_argument("--obs_dim", default=150, type=int)
    parser.add_argument("--action_dim", default=1, type=int)
    parser.add_argument("--buffer_size", default=6_538_000, type=int)
    # parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--debug", type=int, default=0.05)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--actor_load_path", type=str, default=None)
    parser.add_argument("--critic_load_path", type=str, default=None)
    parser.add_argument("--policy_batchsize", type=int, default=256)
    parser.add_argument("--actor_blocks", type=int, default=3)
    parser.add_argument("--z_noise", type=int, default=1)
    parser.add_argument("--WT", type=str, default="VDS")
    parser.add_argument("--WT_noise", type=str, default="False")
    parser.add_argument("--Guassian_act", action="store_true")
    parser.add_argument("--use_adv", action="store_true")
    parser.add_argument("--q_layer", type=int, default=2)
    parser.add_argument("--n_policy_epochs", type=int, default=100)
    parser.add_argument("--policy_layer", type=int, default=4)
    parser.add_argument("--critic_load_epochs", type=int, default=150)
    parser.add_argument("--regq", type=int, default=0)
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch = 1
        args.critic_epoch = 1

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']
    return args