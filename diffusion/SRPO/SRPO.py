import copy, pdb

import torch
import torch.nn as nn

from diffusion.SRPO.model import *


class SRPO(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(
            input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args
        )
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_behavior.parameters(), lr=3e-4
        )
        if args.Guassian_act:
            self.SRPO_policy = GaussianPolicy(output_dim, input_dim - output_dim).to(
                args.device
            )
        else:
            self.SRPO_policy = Dirac_Policy(output_dim, input_dim - output_dim).to(
                args.device
            )
        self.SRPO_policy_optimizer = torch.optim.Adam(
            self.SRPO_policy.parameters(), lr=3e-4
        )
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.SRPO_policy_optimizer, T_max=args.max_steps, eta_min=0.0
        )

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(
            IQL_Critic(adim=output_dim, sdim=input_dim - output_dim, args=args)
        )

    def update_SRPO_policy(self, batch):
        (
            s,
            _,
            _,
            _,
            _,
        ) = batch
        self.diffusion_behavior.eval()
        if self.args.Guassian_act:
            out_, _, _ = self.SRPO_policy(s, torch.zeros((1, 1)), torch.zeros((1, 1)))
            out_ = torch.squeeze(out_, 0)
            mean = out_[:, :1]
            mean = mean / 1e6
            # std = out_[0, 1:]
            # eps = torch.randn_like(mean)
            a = mean  # + std * eps
        else:
            a = self.SRPO_policy(s) / 1e6
        score_a = a
        t = torch.rand(score_a.shape[0], device=s.device) * 0.96 + 0.02  # u(0.02, 0.98)
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(score_a)
        perturbed_a = score_a * alpha_t[..., None] + z * std[..., None]
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.WT_noise:
                episilon = episilon - z
        if "VDS" in self.args.WT:
            wt = std**2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False

        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a, s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)
        v = self.q[0].vf(s)

        adv = q - v
        # TODO be aware that there is a small std gap term here, this seem won't affect final performance though
        q_ = adv if self.args.use_adv else q
        guidance = (
            torch.autograd.grad(torch.sum(q_), detach_a)[0].detach() * std[..., None]
        )
        # guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        if self.args.regq:
            guidance_norm = torch.mean(guidance**2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm
        try:
            loss_1 = (episilon * a).sum(-1) * wt
            loss_2 = (guidance * a).sum(-1)
            loss = loss_1 - loss_2 * self.args.beta
        except:
            pdb.set_trace()
        loss = loss.mean()
        self.SRPO_policy.loss_eps = loss_1.mean()
        self.SRPO_policy.loss_q = loss_2.mean()
        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()
        self.diffusion_behavior.train()
        return loss


class SRPO_Behavior(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(
            input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args
        )
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_behavior.parameters(), lr=3e-4
        )

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0

    def update_behavior(self, batch):
        self.step += 1
        (
            all_s,
            all_a,
            _,
            _,
            _,
        ) = batch

        # Update diffusion behavior
        self.diffusion_behavior.train()

        # all_a /= 20
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1.0 - 1e-3) + 1e-3
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.diffusion_behavior(perturbed_x, random_t, all_s)
        loss = torch.mean(torch.sum((episilon - z) ** 2, dim=(1,)))
        self.loss = loss

        self.diffusion_optimizer.zero_grad()
        loss.backward()
        self.diffusion_optimizer.step()


class SRPO_IQL(nn.Module):
    def __init__(self, input_dim, output_dim, args=None):
        super().__init__()
        self.deter_policy = Dirac_Policy(output_dim, input_dim - output_dim).to(
            args.device
        )
        self.deter_policy_optimizer = torch.optim.Adam(
            self.deter_policy.parameters(), lr=3e-4
        )
        self.deter_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.deter_policy_optimizer, T_max=args.max_steps, eta_min=0.0
        )

        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(
            IQL_Critic(adim=output_dim, sdim=input_dim - output_dim, args=args)
        )

    def update_iql(self, batch):
        (
            s,
            a,
            _,
            _,
            _,
        ) = batch
        self.q[0].update_q0(batch)

        # evaluate iql policy part, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 3.0
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        policy_out = self.deter_policy(s)
        policy_out = policy_out / 1e6
        bc_losses = torch.sum((policy_out - a) ** 2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss


def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        print(args.q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.vf = ValueFunction(sdim).to(args.device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.discount = 0.99
        self.args = args
        self.tau = 0.7
        print(self.tau)
    
    def update_q0(self, batch):
        (
            s,
            a,
            r,
            s_,
            d,
        ) = batch
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        # targets = r + (1.0 - d.float()) * self.discount * next_v.detach()
        targets = r + self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)


class IQL_Critic_df(nn.Module):
    def __init__(self, adim, sdim, q_layer, device) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=q_layer).to(device)
        print(q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(device)

        self.vf = ValueFunction(sdim).to(device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.discount = 0.99
        # self.args = args
        self.tau = 0.7
        print(self.tau)

    def update_q0(self, batch):
        (
            s,
            a,
            r,
            s_,
            d,
        ) = batch
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = r + (1.0 - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)


class IQL_Critic_dt(nn.Module):
    def __init__(self, adim, sdim, q_layer, device) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=q_layer).to(device)
        print(q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(device)

        self.vf = ValueFunction(sdim).to(device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.discount = 0.99
        self.tau = 0.7
        print(self.tau)

    def update_q0(self, batch):
        (
            s,
            a,
            r,
            s_,
            d,
        ) = batch
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        # targets = r + (1.0 - d.float()) * self.discount * next_v.detach()
        targets = r + self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)
