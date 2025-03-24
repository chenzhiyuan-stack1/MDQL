import argparse

# import d4rl
# import gym
import numpy as np
import torch
import random

temperature_coefficients = {
    "antmaze-medium-play-v2": 0.08,
    "antmaze-umaze-v2": 0.02,
    "antmaze-umaze-diverse-v2": 0.04,
    "antmaze-medium-diverse-v2": 0.05,
    "antmaze-large-diverse-v2": 0.05,
    "antmaze-large-play-v2": 0.06,
    "hopper-medium-expert-v2": 0.01,
    "hopper-medium-v2": 0.05,
    "hopper-medium-replay-v2": 0.2,
    "walker2d-medium-expert-v2": 0.1,
    "walker2d-medium-v2": 0.05,
    "walker2d-medium-replay-v2": 0.5,
    "halfcheetah-medium-expert-v2": 0.01,
    "halfcheetah-medium-v2": 0.2,
    "halfcheetah-medium-replay-v2": 0.2,
}


def marginal_prob_std(t, device="cuda", beta_1=20.0, beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""
    t = (
        torch.tensor(t, device=f"cuda:{device[0]}")
        if isinstance(device, list)
        else torch.tensor(t, device=device)
    )
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return alpha_t, std


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = "cpu"
    return device


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="rtc_srpo")  # OpenAI gym environment name
    parser.add_argument(
        "--dataset_path",
        default="/home/min414/data1/Schaferct/training_dataset_pickle/v8.pickle",
    )
    parser.add_argument(
        "--add_data_path",
        default=None,
    )
    parser.add_argument(
        "--eval_data_path",
        default="/home/min414/data1/Schaferct/ALLdatasets/1",
    )
    parser.add_argument(
        "--ckpt_path",
        default="/home/min414/data1/offlineRL-rtc-bwp/rtc_srpo/checkpoints",
    )
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds eval_interval
    parser.add_argument("--eval_num", default=8, type=int)
    parser.add_argument("--max_steps", default=int(2e6), type=int)
    parser.add_argument("--eval_interval", default=int(1e5), type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--save_model", default=1, type=int)
    parser.add_argument("--obs_dim", default=150, type=int)
    parser.add_argument("--action_dim", default=1, type=int)
    parser.add_argument("--buffer_size", default=6_538_000, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
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
    print("**************************")
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch = 1
        args.critic_epoch = 1
    print(args)
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
