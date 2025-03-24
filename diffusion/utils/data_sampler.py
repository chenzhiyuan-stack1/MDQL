# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np

# from v14_iql.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
TensorBatch = List[torch.Tensor]
b_in_Mb = 1e6

class Data_Sampler(object):
	def __init__(self, data, device, reward_tune='no'):
		
		self.state = torch.from_numpy(data['observations']).float()
		self.action = torch.from_numpy(data['actions']).float()
		self.next_state = torch.from_numpy(data['next_observations']).float()
		reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
		self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]

		self.device = device

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))

		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)


def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        print(f"Dataset size: {n_transitions}")
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset loaded: {self._pointer}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        # states = torch.unsqueeze(states, 0)
        return [states.to(self._device), actions.to(self._device), rewards.to(self._device), next_states.to(self._device), dones.to(self._device)]

    def add_transition(self, data: Dict[str, np.ndarray]):
        # raise NotImplementedError
        if self._size != 0:
            # raise ValueError("Trying to load data into non-empty replay buffer")
            ini_idx = self._pointer
        else:
            ini_idx = 0
        n_transitions = data["observations"].shape[0]
        print(f"Dataset size add: {n_transitions}")
        if n_transitions > (self._buffer_size - ini_idx):
            raise ValueError(
                f"Replay buffer is {ini_idx + n_transitions - self._buffer_size} smaller than the dataset you are trying to load!"
            )
        self._states[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["observations"])
        self._actions[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["actions"])
        self._rewards[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, ini_idx + n_transitions)
        print(f"Dataset added: {self._pointer}")