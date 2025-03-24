# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusion.utils.logger import logger

from diffusion.agents.meta_diffusion_id_TD7 import Diffusion
from diffusion.agents.model import MLP
from diffusion.agents.model import MLP_GRU, Meta_MLP_GRU
from diffusion.agents.helpers import EMA

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import functools

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

        self.model = Meta_MLP_GRU(state_dim=state_dim, action_dim=action_dim, z_dim=256 , device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, eta=eta, args=args,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        # import TD7 learned critic and avoid training it
        self.v_critic = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/v_critic_1182.0.pt').to(args.device).eval()
        self.q_critic = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/q_critic_1182.0.pt').to(args.device).eval()
        self.encoder = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/encoder_1182.0.pt').to(args.device).eval()
        # self.v_critic = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/1116/v_critic_220.0.pt').to(args.device).eval()
        # self.q_critic = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/1116/q_critic_220.0.pt').to(args.device).eval()
        # self.encoder = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/1116/encoder_220.0.pt').to(args.device).eval()
        for param in self.v_critic.parameters():
            param.requires_grad = False
        for param in self.q_critic.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # action (Mbps)
            state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)


            """ Latent variable inference """
            with torch.no_grad():
                zs = self.encoder.zs(state)
                zsa = self.encoder.zsa(zs, action)
                
            
            """ Policy Training """
            bc_loss = self.actor.loss(action, state, zs, zsa)
            actor_loss = bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()

        return metric

    def sample_action(self, state, z):
        state = torch.FloatTensor(state.to("cpu").reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        z_rpt = torch.repeat_interleave(z.to(self.device), repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt, z_rpt)
            zsa_rpt = self.encoder.zsa(z_rpt, action / 1e6)
            q_value = self.q_critic(state_rpt, action / 1e6, zsa_rpt, z_rpt).flatten()
            q_value = torch.clamp(q_value, min=0)
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
