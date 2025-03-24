import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, "MLP requires at least two dims (input and output)"
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, layers=2):
        super().__init__()
        dims = [state_dim + action_dim] + [256] * layers + [1]
        # dims = [state_dim + action_dim, 256, 256, 1] # TODO
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        # print(action.shape, condition.shape)
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        act_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = 20

        # encoder 1
        self.encoder1 = nn.Sequential(
            # encoder 1
            nn.Linear(state_dim, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        # Recisual Block 1(rb1)
        self.rb1 = nn.Sequential(
            nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU()
        )
        # Recisual Block 2(rb2)
        self.rb2 = nn.Sequential(
            nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU()
        )
        # final 'gmm'
        self.final = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, obs: torch.Tensor, h, c):
        obs_ = torch.squeeze(obs, 0)
        ###
        mean = self.encoder1(obs_)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean)
        ###
        mean = mean * self.max_action * 1e6  # Mbps -> bps
        mean = mean.clamp(min=10)  # larger than 10bps
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c
    
    def select_actions(self, obs: torch.Tensor, h, c):
        obs_ = torch.squeeze(obs, 0)
        ###
        mean = self.encoder1(obs_)
        mean, _ = self.gru(mean)
        mean = self.fc_mid(mean)
        mem1 = mean
        mean = self.rb1(mean) + mem1
        mem2 = mean
        mean = self.rb2(mean) + mem2
        mean = self.final(mean)
        ###
        mean = mean * self.max_action * 1e6  # Mbps -> bps
        mean = mean.clamp(min=10)  # larger than 10bps
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(mean.shape[0], 1)
        ret = torch.cat((mean, std), 1)
        ret = torch.unsqueeze(ret, 0)  # (1, bs, 2)
        return ret, h, c

class Dirac_Policy(nn.Module):

    def __init__(
        self,
        action_dim,
        state_dim,
        dropout_rate=None,
        num_blocks=3,
        use_layer_norm=False,
        hidden_dim=256,
        activations=nn.Mish(),
    ):
        super(Dirac_Policy, self).__init__()
        # self.net = mlp(
        #     [state_dim] + [256] * layer + [action_dim], output_activation=nn.Tanh
        # )
        self.num_blocks = num_blocks
        self.out_dim = action_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations
        # self.activation_outputs = nn.Tanh()

        self.fc = nn.Linear(state_dim, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    self.hidden_dim,
                    self.activations,
                    self.dropout_rate,
                    self.use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid()
        )
        # self.out_fc = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.out_dim), nn.Tanh()
        # )

    def forward(self, state):
        x = self.fc(state)

        for block in self.blocks:
            x = block(x, training=False)

        x = self.activations(x)
        mean = self.out_fc(x)
        mean = mean * 20 * 1e6  # max action value
        mean = mean.clamp(min=10)
        return mean

    def select_actions(self, state):
        x = self.fc(state)

        for block in self.blocks:
            x = block(x, training=False)

        x = self.activations(x)
        mean = self.out_fc(x)
        mean = mean * 20 * 1e6  # max action value
        mean = mean.clamp(min=10)
        return mean


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""

    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate is not None and dropout_rate > 0.0
            else None
        )

    def forward(self, x, training=False):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        input_dim,
        out_dim,
        dropout_rate=None,
        use_layer_norm=False,
        hidden_dim=256,
        activations=F.relu,
    ):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim + 128, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    self.hidden_dim,
                    self.activations,
                    self.dropout_rate,
                    self.use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x


class ScoreNet_IDQL(nn.Module):
    def __init__(
        self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device = args.device
        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.main = MLPResNet(
            args.actor_blocks,
            input_dim,
            output_dim,
            dropout_rate=0.1,
            use_layer_norm=True,
            hidden_dim=256,
            activations=nn.Mish(),
        )
        self.cond_model = mlp(
            [64, 128, 128], output_activation=None, activation=nn.Mish
        )

        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t, condition):
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, condition, embed], dim=-1)
        h = self.main(all)
        return h
