# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        # print(x.shape, t.shape, state.shape)
        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度
        # print(x.shape, t.shape, state.shape)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class MLP_GRU(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_GRU, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # use GRU to encode the state
        # encoder 1
        self.encoder1 = nn.Sequential(
            # encoder 1
            nn.Linear(state_dim, 256),
            # nn.LayerNorm(256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
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
            nn.Linear(256, 10),
            nn.Tanh()
        )
        
        input_dim = action_dim + 10 + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        
        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度
        # print(x.shape, t.shape, state.shape)
        mean = self.encoder1(state)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean) # Mbps
        # mean = mean * self.max_action * 1e6  # Mbps -> bps
        # mean = mean.clamp(min = 10)  # larger than 10bps
        
        
        # x = torch.cat([x, t, state], dim=1)
        x = torch.cat([x, t, mean], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class Meta_MLP_GRU(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim = 150,
                 action_dim = 1,
                 z_dim = 256,
                 device = 'cpu',
                 t_dim = 16):

        super(Meta_MLP_GRU, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # use GRU to encode the state
        # encoder 1
        self.encoder1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
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
            nn.Linear(256, 10),
            nn.Tanh()
        )
        
        input_dim = action_dim + 10 + t_dim + z_dim
        self.layernorm = nn.LayerNorm(input_dim)
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        
        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度
        
        mean = self.encoder1(state)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean) # Mbps
        x = torch.cat([x, t, mean, z], dim=1)
        x = self.layernorm(x)
        x = self.mid_layer(x)

        return self.final_layer(x)

class Meta_MLP_GRU_v1(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim = 150,
                 action_dim = 1,
                 z_dim = 256,
                 device = 'cpu',
                 t_dim = 16):

        super(Meta_MLP_GRU_v1, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # use GRU to encode the state
        # encoder 1
        self.encoder1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        # GRU
        # self.gru = nn.GRU(256, 256, 2)
        self.gru_z_l0 = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())
        self.gru_h_l0 = nn.Linear(256, 256)
        self.fc_l0 = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        self.gru_z_l1 = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())
        self.gru_h_l1 = nn.Linear(256, 256)
        self.fc_l1 = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
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
            nn.Linear(256, 10),
            nn.Tanh()
        )
        
        input_dim = action_dim + 10 + t_dim + z_dim
        self.layernorm = nn.LayerNorm(input_dim)
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        
        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度
        
        mean = self.encoder1(state)
        # mean, _ = self.gru(mean)
        
        z_l0 = self.gru_z_l0(mean)
        h_l0 = self.gru_h_l0(mean)
        emb = self.fc_l0(z_l0 * h_l0)
        z_l1 = self.gru_z_l1(emb)
        h_l1 = self.gru_h_l1(emb)
        mean = self.fc_l1(z_l1 * h_l1)
        
        
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean) # Mbps
        
        x = torch.cat([x, t, mean, z], dim=1)
        x = self.layernorm(x)
        x = self.mid_layer(x)

        return self.final_layer(x)

class Meta_MLP_GRU_select_feature(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim = 150,
                 action_dim = 1,
                 z_dim = 256,
                 device = 'cpu',
                 t_dim = 16):

        super(Meta_MLP_GRU_select_feature, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # use GRU to encode the state
        # encoder 1
        self.encoder1 = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
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
            nn.Linear(256, 10),
            nn.Tanh()
        )
        
        input_dim = action_dim + 10 + t_dim + z_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)
        
        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度
        
        # select features
        state = state.view(-1, 15, 10)
        indices = torch.tensor([0, 1, 3, 10, 11]).to("cuda:0")
        # indices = torch.tensor([0, 1, 3, 10, 11])
        state = torch.index_select(state, 1, indices)
        state = state.view(-1, 50)
        
        mean = self.encoder1(state)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean) # Mbps
        
        x = torch.cat([x, t, mean, z], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
    
class MLP_GRU_select_feature(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_GRU_select_feature, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # use GRU to encode the state
        # encoder 1
        self.encoder1 = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
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
            nn.Linear(256, 10),
            nn.Tanh()
        )

        input_dim = action_dim + 10 + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state, z):

        t = self.time_mlp(time)

        if state.dim() == 3:
            state = torch.squeeze(state, 0)  # 假设要去掉的是第一个维度

        state1 = state.view(-1, 15, 10)
        indices = torch.tensor([0, 1, 3, 10, 11]).to("cuda:0")
        # indices = torch.tensor([0, 1, 3, 10, 11])
        state1 = torch.index_select(state1, 1, indices)
        state1 = state1.view(-1, 50)

        mean = self.encoder1(state1)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean) # Mbps
        
        x = torch.cat([x, t, mean], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)