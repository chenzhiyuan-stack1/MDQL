# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion.agents.meta_diffusion_id_TD7_together import Diffusion
from diffusion.agents.model import MLP
from diffusion.agents.model import MLP_GRU, Meta_MLP_GRU
from diffusion.agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

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
    # zs_dim: int = 256
    zs_dim: int = 10
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

def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
		return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Encoder_select_feature(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder_select_feature, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(50, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)
	

	def zs(self, state):
		state = state.view(-1, 15, 10)
		indices = torch.tensor([0, 1, 3, 10, 11]).to("cuda:0")
		# indices = torch.tensor([0, 1, 3, 10, 11])
		state = torch.index_select(state, 1, indices)
		state = state.view(-1, 50)

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


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 args=None,
                 ):

        self.model = Meta_MLP_GRU(state_dim=state_dim, action_dim=action_dim, z_dim=10 , device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, eta=eta, args=args,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        # train TD7 together
        hp = Hyperparameters()
        self.hp = hp
        self.q_critic = Q_Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=hp.critic_lr)
        self.q_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.q_critic_optimizer, T_max=lr_maxt, eta_min=0.0)
        self.q_critic_target = copy.deepcopy(self.q_critic).requires_grad_(False).to(device)

        self.v_critic = V_Critic(state_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(device)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=hp.critic_lr)
        self.v_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.v_critic_optimizer, T_max=lr_maxt, eta_min=0.0)

        # self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(device)
        self.encoder = Encoder_select_feature(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=lr_maxt, eta_min=0.0)
        self.fixed_encoder = copy.deepcopy(self.encoder).requires_grad_(False).to(device) # t-1
        self.fixed_encoder_target = copy.deepcopy(self.encoder).requires_grad_(False).to(device) # t-2

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        
        # Value clipping tracked values in TD7
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'actor_loss': []}
        for _ in range(iterations):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
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

            log_writer.log({"encoder_loss": encoder_loss.item()}, step=self.step)
            log_writer.log({"lr": self.encoder_optimizer.state_dict()["param_groups"][0]["lr"]}, step=self.step)

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
            v_loss = asymmetric_l2_loss(adv, 0.7) # log
            log_writer.log({"v_loss": v_loss.item()}, step=self.step)
            log_writer.log({"v": V.mean()}, step=self.step)
            self.v_critic_optimizer.zero_grad()
            v_loss.backward()
            self.v_critic_optimizer.step()
            self.v_critic_lr_scheduler.step()

            # update Q
            target = reward + (1.0 - done.float()) * self.hp.discount * next_v.detach()
            Q = self.q_critic.both(state, action, fixed_zsa, fixed_zs)
            td_loss = ((Q[0] - target).abs() + (Q[1] - target).abs()) / 2.
            td_loss_1 = (Q[0] - target).abs()
            td_loss_2 = (Q[1] - target).abs()

            # using Q and target to calculate the q_loss, different choices of q_loss
            q_loss = LAP_huber(td_loss)
            # q_loss = (LAP_huber(td_loss_1) + LAP_huber(td_loss_2)) / 2.
            # q_loss = sum(F.mse_loss(q, target) for q in Q) / len(Q)
            # q_loss = (F.mse_loss(Q[0], target) + F.mse_loss(Q[1], target)) / 2.


            log_writer.log({"q_loss": q_loss.item()}, step=self.step)
            log_writer.log({"q": Q[0].mean()}, step=self.step)
            log_writer.log({"q_target": target.mean()}, step=self.step)
            log_writer.log({"td_loss": td_loss.mean()}, step=self.step)
            self.q_critic_optimizer.zero_grad()
            q_loss.backward()
            self.q_critic_optimizer.step()
            self.q_critic_lr_scheduler.step()
            
            
            #########################
            # Update Actor
            #########################
            """ Latent variable inference """
            with torch.no_grad():
                zs = self.fixed_encoder.zs(state)
            
            """ Policy Training """
            bc_loss = self.actor.loss(action, state, zs.detach(), adv.detach(), Q_target.detach())
            actor_loss = bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            
            """ Log """
            if self.grad_norm > 0:
                log_writer.log({'Actor Grad Norm': actor_grad_norms.max().item()}, step = self.step)
            log_writer.log({'BC Loss': bc_loss.item()}, step = self.step)

            #########################
            # Update Iteration
            #########################
            if self.step % self.hp.target_update_rate == 0:
                self.q_critic_target.load_state_dict(self.q_critic.state_dict())
                self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
                self.fixed_encoder.load_state_dict(self.encoder.state_dict())
                
                # self.replay_buffer.reset_max_priority()

                self.max_target = self.max
                self.min_target = self.min
                            
            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()

        return metric

    def sample_action(self, state, fixed_target_zs):
        state = torch.FloatTensor(state.to("cpu").reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        fixed_target_zs_rpt = torch.repeat_interleave(fixed_target_zs.to(self.device), repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt, fixed_target_zs_rpt)
            fixed_target_zsa_rpt = self.fixed_encoder_target.zsa(fixed_target_zs_rpt, action / 1e6)
            q_value = self.q_critic_target(state_rpt, action / 1e6, fixed_target_zsa_rpt, fixed_target_zs_rpt).flatten()
            q_value = torch.clamp(q_value, min=0)
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.encoder.state_dict(), f'{dir}/encoder_{id}.pth')
            torch.save(self.q_critic.state_dict(), f'{dir}/q_critic_{id}.pth')
            torch.save(self.v_critic.state_dict(), f'{dir}/v_critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
