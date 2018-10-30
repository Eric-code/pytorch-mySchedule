import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer
import numpy as np


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.old_action_log_probs = torch.Tensor([8000, 1])

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        old_values = rollouts.value_preds[:-1].view(num_steps, num_processes, 1)

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        values = values.view(num_steps, num_processes, 1)
        vpredclipped = old_values + np.clip((values - old_values).detach(), - 0.2, 0.2)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        self.old_action_log_probs = action_log_probs

        # modified loss:
        # advantages1 = (rollouts.returns[:-1] - values).detach().pow(2)
        # advantages2 = (rollouts.returns[:-1] - vpredclipped).detach().pow(2)
        # advantages = np.maximum(advantages1, advantages2)
        # value_loss = advantages.mean()
        #
        # # advantages = rollouts.returns[:-1] - values
        # # value_loss = advantages.pow(2).mean()
        #
        # ratio = torch.Tensor.exp(self.old_action_log_probs - action_log_probs)
        # # action_loss1 = -(advantages.detach() * action_log_probs * ratio)
        # ratio = np.clip(ratio.detach(), 0.8, 1.2)
        # # action_loss2 = -(advantages.detach() * action_log_probs * ratio)
        # # action_loss = np.maximum(action_loss1.detach(), action_loss2.detach()).mean()
        # ppd = (self.old_action_log_probs - action_log_probs).pow(2).mean()
        # # action_loss = action_loss1.detach().mean()
        #
        # action_loss = -(advantages.detach() * action_log_probs * ratio).mean()

        # original loss:
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        # (value_loss * self.value_loss_coef + action_loss -
        #  dist_entropy * self.entropy_coef + 0.5 * ppd).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
