# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import os
import torch
import json
import math

# import d4rl
from diffusion.utils import utils
from diffusion.utils.data_sampler import Data_Sampler
from diffusion.utils.logger import logger, setup_logger

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
import sys

hyperparameters = {
    'meta-v14-id-gru-eta10-TD7-onlyzs-good33':     {'lr': 1e-5, 'eta': 10.0,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 10, 'num_epochs': 4000, 'gn': 10.0, 'top_k': 1},
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
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = './checkpoints_diffusionql'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL 的参数，部分可以用作 Diffusion QL 的参数
    # buffer_size: int = 6_538_000  # Replay buffer size
    buffer_size: int = 20_000_000  # Replay buffer size
    # buffer_size: int = 8711389  # Replay buffer size
    batch_size: int = 2048  # Batch size for all networks
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

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def adjust_dataset(dataset_dict):
    dataset_dict["actions"] /= b_in_Mb

    normal_vector = NORMAL_VECTOR
    
    dataset_dict["observations"] *= normal_vector
    dataset_dict["next_observations"] *= normal_vector

    return dataset_dict

def evaluate_RtcBwp(policy_fn, eval_dataset: list, device: str):
    every_call_mse = []
    every_call_accuracy = []
    every_call_over = []
    for f_path in tqdm(eval_dataset, desc="Evaluating"):
        with open(f_path, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data["observations"], dtype=np.float32)
        normal_vector = NORMAL_VECTOR
        true_capacity = np.asarray(call_data["true_capacity"], dtype=np.float32)

        actions = np.asarray(call_data["bandwidth_predictions"], dtype=np.float32)
        
        quality_videos = np.asarray(call_data["video_quality"], dtype=np.float32)
        quality_audios = np.asarray(call_data["audio_quality"], dtype=np.float32)
        avg_q_v = np.nanmean(quality_videos)
        avg_q_a = np.nanmean(quality_audios)
        rewards = []
        next_observations = []
        for idx in range(observations.shape[0]):
            r_v = quality_videos[idx]
            r_a = quality_audios[idx]
            if math.isnan(quality_videos[idx]):
                r_v = avg_q_v
            if math.isnan(quality_audios[idx]):
                r_a = avg_q_a
            rewards.append(r_v * 1.8 + r_a * 0.2)

            if idx + 1 >= observations.shape[0]:
                next_observations.append([-1] * 150)  # s_terminal
            else:
                next_observations.append(observations[idx + 1])
                
        rewards = np.asarray(rewards, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        
        model_predictions = []
        for t in range(observations.shape[0]):
            obss = observations[t : t + 1, :].reshape(-1)
            obss_ = obss * normal_vector
            obss_ = torch.tensor(obss_.reshape(1, -1), device=device, dtype=torch.float32)
            action = actions[t:t+1].reshape(-1)
            action = action / 1e6
            action_tensor = torch.tensor(action.reshape(1, -1), dtype=torch.float32).to(device)
            reward = rewards[t:t+1].reshape(-1)
            reward_tensor = torch.tensor(reward.reshape(1, -1), dtype=torch.float32).to(device)
            next_obss = next_observations[t:t+1, :].reshape(-1)
            next_obss = next_obss * normal_vector
            next_obss_tensor = torch.tensor(next_obss.reshape(1, -1), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action = policy_fn.sample_action(obss_) # bps
                action = action / 1e6
            bw_prediction = np.squeeze(action)
            try:
                assert np.size(bw_prediction) == 1 
            except:
                pdb.set_trace()
            model_predictions.append(bw_prediction)
        # mse and accuracy of this call
        model_predictions = np.asarray(model_predictions, dtype=np.float32)
        true_capacity = true_capacity / 1e6
        # model_predictions = model_predictions / 1e6
        call_mse = []
        call_accuracy = []
        call_over = []
        for true_bw, pre_bw in zip(true_capacity, model_predictions):
            if np.isnan(true_bw) or np.isnan(pre_bw):
                continue
            else:
                mse_ = (true_bw - pre_bw) ** 2
                call_mse.append(mse_)
                accuracy_ = max(0, 1 - abs(pre_bw - true_bw) / true_bw)
                call_accuracy.append(accuracy_)
                over = max(0,(pre_bw - true_bw) / true_bw)
                call_over.append(over)
        call_mse = np.asarray(call_mse, dtype=np.float32)
        every_call_mse.append(np.mean(call_mse))
        call_accuracy = np.asarray(call_accuracy, dtype=np.float32)
        every_call_accuracy.append(np.mean(call_accuracy))
        call_over = np.asarray(call_over, dtype=np.float32)
        every_call_over.append(np.mean(call_over))
    every_call_mse = np.asarray(every_call_mse, dtype=np.float32)
    every_call_accuracy = np.asarray(every_call_accuracy, dtype=np.float32)
    every_call_over = np.asarray(every_call_over, dtype=np.float32)
    return np.mean(every_call_mse), np.mean(every_call_accuracy), np.mean(every_call_over)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
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
        default = "/home/czy/Schaferct/mstrain-id-123.pickle",
    )
    parser.add_argument(
        "--add_data_path",
        # default="/home/min414/data3/mstrain-id-345.pickle",
        default="/home/czy/Schaferct/mstrain-id-345.pickle",
    )
    parser.add_argument(
        "--eval_data_path",
        default="/home/czy/Schaferct/ALLdatasets/emulate",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
    ) 
    parser.add_argument(
        "--critic_path",
        default="/home/czy/meta/Meta_diffusion_QL/diffusion/SRPO/critic_ckpt120000.pth",
    )
    # Sets Gym, PyTorch and Numpy seeds eval_interval
    parser.add_argument("--eval_num", default=8, type=int)
    parser.add_argument("--max_steps", default=int(2e6), type=int)
    parser.add_argument("--eval_interval", default=int(1e5), type=int)
    # parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--save_model", default=1, type=int)
    parser.add_argument("--obs_dim", default=150, type=int)
    parser.add_argument("--action_dim", default=1, type=int)
    # parser.add_argument("--buffer_size", default=6_538_000, type=int)
    parser.add_argument("--buffer_size", default=20_000_000, type=int)
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

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")

    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    
    # Load dataset for this experiment
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    from diffusion.utils.data_sampler import ReplayBuffer
    config = TrainConfig()
    
    
    testdataset_file = open(args.dataset_path, 'rb')
    dataset = pickle.load(testdataset_file)
    # Normalize the actions and states
    dataset = adjust_dataset(dataset)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        # config.device,
    )
    replay_buffer.load_dataset(dataset)
    del dataset
    if args.add_data_path is not None:
        testdataset_file_add = open(args.add_data_path, 'rb')
        data_add = pickle.load(testdataset_file_add)
        # Normalize the actions and states
        data_add = adjust_dataset(data_add)
        replay_buffer.add_transition(data_add)
        del data_add
    
    # replay_buffer = ReplayBuffer(
    #     state_dim,
    #     action_dim,
    #     40_000_000,
    #     # config.device,
    # )
    # for i in range(6):
    #     data_path = f'/data/Schaferct/training_dataset_pickle/mstrain-num-1600-id-{i}.pickle'
    #     dataset = pickle.load(open(data_path, 'rb'))
    #     dataset = adjust_dataset(dataset)
    #     replay_buffer.add_transition(dataset)
    #     del dataset
    
    print('dataset loaded')
    
    

    max_action = MAX_ACTION
    
    # Set seeds
    seed = config.seed
    set_seed(seed)
    
    print("---------------------------------------")
    print(f"Training Diffusion-QL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    
    from diffusion.agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=config.device,
                    discount=args.discount,
                    tau=args.tau,
                    max_q_backup=args.max_q_backup,
                    beta_schedule=args.beta_schedule,
                    n_timesteps=args.T,
                    eta=args.eta,
                    lr=args.lr,
                    lr_decay=args.lr_decay,
                    lr_maxt=args.num_epochs,
                    grad_norm=args.gn,
                    args = args)
    
    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(replay_buffer,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.dump_tabular()

        # Evaluate episode
        print(f"Time steps: {training_iters + 1}")
        mse_, accuracy_, over_ = evaluate_RtcBwp(agent, small_evaluation_datasets, args.device)
        evaluations.append([mse_, accuracy_, over_, 
                            np.mean(loss_metric['bc_loss']),
                            np.mean(loss_metric['actor_loss']),
                            curr_epoch])
        np.save(os.path.join(results_dir, "eval"), evaluations)
        logger.record_tabular('mse_', mse_)
        logger.record_tabular('accuracy_', accuracy_)
        logger.record_tabular('over_', over_)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss
        agent.save_model(results_dir, curr_epoch)
            
    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 0])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best mse': scores[best_id, 0],
                    'best accuracy': scores[best_id, 1]}
        with open(os.path.join(results_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 2]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best mse': scores[where_k][0][0],
                    'best accuracy': scores[where_k][0][1]}

        with open(os.path.join(results_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
