import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 1024
	discount: float = 0.99
	target_update_rate: int = 250
	
	# TD3
	policy_freq: int = 2
	
	# LAP
	alpha: float = 0.4
	min_priority: float = 1
	
	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-6
	
	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-6
	
	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = F.relu
	actor_lr: float = 3e-4


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)
		

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		a = self.l3(a)
		a = torch.tanh(a)
		a = a * 20 * 1e6 # max 20 Mbps
		return a


class Actor_resiual(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor_resiual, self).__init__()

		self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
		# encoder 1
		self.encoder1 = nn.Sequential(
			nn.Linear(150, 256),
			nn.ReLU()
		)
		# GRU
		self.gru = nn.GRU(256, 256, 2)
		# FC
		self.fc_mid = nn.Sequential(
			nn.Linear(256 + 256, 256),
			nn.ReLU()
		)
		# Recisual Block 1(rb1)
		self.rb1 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 256),
			nn.LeakyReLU()
		)
		# Recisual Block 2(rb2)
		self.rb2 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 256),
			nn.LeakyReLU()
		)
		# final 'gmm'
		self.final = nn.Sequential(
			nn.Linear(256, 1),
			nn.Tanh()
		)
		

	def forward(self, state, zs):
		a = self.encoder1(state)
		a = AvgL1Norm(a)
		a = torch.cat([a, zs], 1)
		a = self.fc_mid(a)
		mem1 = a
		a = self.rb1(a) + mem1
		mem2 = a
		a = self.rb2(a) + mem2
		a = self.final(a)
		a = a * 20 * 1e6 # max 20 Mbps
  
		# a = AvgL1Norm(self.l0(state))
		# a = torch.cat([a, zs], 1)
		# a = self.activ(self.l1(a))
		# a = self.activ(self.l2(a))
		# a = self.l3(a)
		# a = torch.tanh(a)
		# a = a * 20 * 1e6 # max 20 Mbps
		return a


class Actor_resiual_gru(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor_resiual_gru, self).__init__()

		self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
		# encoder 1
		self.encoder1 = nn.Sequential(
			nn.Linear(150, 256),
			nn.ReLU()
		)
		# GRU
		self.gru = nn.GRU(256, 256, 2)
		# FC
		self.fc_mid = nn.Sequential(
			nn.Linear(256 + 256, 256),
			nn.ReLU()
		)
		# Recisual Block 1(rb1)
		self.rb1 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 256),
			nn.LeakyReLU()
		)
		# Recisual Block 2(rb2)
		self.rb2 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 256),
			nn.LeakyReLU()
		)
		# final 'gmm'
		self.final = nn.Sequential(
			nn.Linear(256, 1),
			nn.Tanh()
		)
		

	def forward(self, state, zs):
		a = self.encoder1(state)
		a = a.unsqueeze(0)
		a, _ = self.gru(a)
		a = a.squeeze(0)
		a = AvgL1Norm(a)
		a = torch.cat([a, zs], 1)
		a = self.fc_mid(a)
		mem1 = a
		a = self.rb1(a) + mem1
		mem2 = a
		a = self.rb2(a) + mem2
		a = self.final(a)
		a = a * 20 * 1e6 # max 20 Mbps
		return a


class Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)
	

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs


	def zsa(self, zs, action):
		zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa


class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Q_Critic, self).__init__()

		self.activ = activ
		
		self.q01 = nn.Linear(state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def both(self, state, action, zsa, zs):
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sa))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return q1, q2

	def forward(self, state, action, zsa, zs):
		return torch.min(*self.both(state, action, zsa, zs))


class V_Critic(nn.Module):
	def __init__(self, state_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(V_Critic, self).__init__()

		self.activ = activ
		
		self.v01 = nn.Linear(state_dim, hdim)
		self.v1 = nn.Linear(zs_dim + hdim, hdim)
		self.v2 = nn.Linear(hdim, hdim)
		self.v3 = nn.Linear(hdim, 1)


	def forward(self, state, zs):
		s = state
		embeddings = zs

		v1 = AvgL1Norm(self.v01(s))
		v1 = torch.cat([v1, embeddings], 1)
		v1 = self.activ(self.v1(v1))
		v1 = self.activ(self.v2(v1))
		v1 = self.v3(v1)

		return v1


class IQL_Agent(object):
	def __init__(self, state_dim, action_dim, replay_buffer, hp=Hyperparameters()): 
		# Changing hyperparameters example: hp=Hyperparameters(batch_size=128)
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.hp = hp

		# self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		# self.actor = Actor_resiual(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor = Actor_resiual_gru(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=1e7, eta_min=0.0)

		self.q_critic = Q_Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=hp.critic_lr)
		self.q_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.q_critic_optimizer, T_max=1e7, eta_min=0.0)
		self.q_critic_target = copy.deepcopy(self.q_critic).requires_grad_(False).to(self.device)
		
		self.v_critic = V_Critic(state_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=hp.critic_lr)
		self.v_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.v_critic_optimizer, T_max=1e7, eta_min=0.0)

		self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.encoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=1e7, eta_min=0.0)
		self.fixed_encoder = copy.deepcopy(self.encoder).requires_grad_(False).to(self.device) # t-1
		self.fixed_encoder_target = copy.deepcopy(self.encoder).requires_grad_(False).to(self.device) # t-2

		self.replay_buffer = replay_buffer

		self.training_steps = 0

		# Value clipping tracked values
		self.max = -1e8
		self.min = 1e8
		self.max_target = 0
		self.min_target = 0
	
	def train(self, args):
		self.training_steps += 1

		# state, action, reward, next_state, done = self.replay_buffer.LAP_sample(self.hp.batch_size)
		state, action, reward, next_state, done = self.replay_buffer.sample(self.hp.batch_size)
		state = state.to(self.device)
		action = action.to(self.device)
		next_state = next_state.to(self.device)
		reward = reward.to(self.device)
		done = done.to(self.device)

		#########################
		# Update Encoder
		#########################
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)

		zs = self.encoder.zs(state)
		pred_zs = self.encoder.zsa(zs, action)
		encoder_loss = F.mse_loss(pred_zs, next_zs)

		args.run.log({"encoder_loss": encoder_loss.item()}, step=self.training_steps)
		args.run.log({"lr": self.encoder_optimizer.state_dict()["param_groups"][0]["lr"]}, step=self.training_steps)

		self.encoder_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()
		self.encoder_lr_scheduler.step()

		#########################
		# Update Critic
		#########################
		# update V
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(state)
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, action)
			
			Q_target = self.q_critic_target(state, action, fixed_target_zsa, fixed_target_zs)
			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))
			
			fixed_zs = self.fixed_encoder.zs(state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

			fixed_next_zs = self.fixed_encoder.zs(next_state)
			
			next_v = self.v_critic(next_state, fixed_next_zs)
		
		V = self.v_critic(state, fixed_zs)
		adv = Q_target - V
		v_loss = self.asymmetric_l2_loss(adv, 0.7) # log
		args.run.log({"v_loss": v_loss.item()}, step=self.training_steps)
		args.run.log({"v": V.mean()}, step=self.training_steps)
		self.v_critic_optimizer.zero_grad()
		v_loss.backward()
		self.v_critic_optimizer.step()
		self.v_critic_lr_scheduler.step()
  
		# update Q
		target = reward + (1.0 - done.float()) * self.hp.discount * next_v.detach()
		Q = self.q_critic.both(state, action, fixed_zsa, fixed_zs)
		td_loss = ((Q[0] - target).abs() + (Q[1] - target).abs()) / 2.

		# using Q and target to calculate the q_loss, different choices of q_loss
		# q_loss = LAP_huber(td_loss)
		q_loss = sum(F.mse_loss(q, target) for q in Q) / len(Q)
		# q_loss = (F.mse_loss(Q[0], target) + F.mse_loss(Q[1], target)) / 2.


		args.run.log({"q_loss": q_loss.item()}, step=self.training_steps)
		args.run.log({"q": Q[0].mean()}, step=self.training_steps)
		args.run.log({"q_target": target.mean()}, step=self.training_steps)
		args.run.log({"td_loss": td_loss.mean()}, step=self.training_steps)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()
		self.q_critic_lr_scheduler.step()
  
		#########################
		# Update LAP
		#########################
		# priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		priority = td_loss.clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)
  
		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			# with torch.no_grad():
			# 	Q_target = self.q_critic_target(state, action, fixed_target_zsa, fixed_target_zs)
			# 	V = self.v_critic(state, fixed_zs)
			# adv = Q_target - V
			exp_adv = torch.exp(3. * adv.detach()).clamp(max=100.0)
			
			policy_out = self.actor(state, fixed_zs) / 1e6
			
			# AWR for Gaussian policy	
			log_std = self.actor.log_std 
			std = torch.exp(log_std) 
			gaussian_policy = Normal(policy_out, std)
			bc_losses = -gaussian_policy.log_prob(action).sum(-1, keepdim=False)
			actor_loss = torch.mean(exp_adv * bc_losses)

			# # AWR for Gaussian policy
			# log_std = self.actor.log_std
			# std = torch.exp(log_std)  # 标准差
			# var = std ** 2  # 方差
			# log_prob = - ((action - policy_out) ** 2) / (2 * var) - log_std - 0.5 * torch.log(2 * torch.tensor(torch.pi))
			# actor_loss = - torch.mean(exp_adv * log_prob) # min loss 

			# # weighted BC for deterministic policy
			# bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
			# actor_loss = torch.mean(exp_adv.squeeze() * bc_losses)

			args.run.log({"actor_loss": actor_loss.item()}, step=self.training_steps)
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.actor_lr_scheduler.step()

		#########################
		# Update Q_target
		#########################
		soft_update(self.q_critic_target, self.q_critic)

		#########################
		# Update Iteration
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			# self.q_critic_target.load_state_dict(self.q_critic.state_dict())
			self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			
			self.replay_buffer.reset_max_priority()

			self.max_target = self.max
			self.min_target = self.min
  
	def asymmetric_l2_loss(self, u: torch.Tensor, tau: float) -> torch.Tensor:
		return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)