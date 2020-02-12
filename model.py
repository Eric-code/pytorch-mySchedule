import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError
        # self.base = MLPBase(obs_shape, **base_kwargs)
        #
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        # self.target = init_(nn.Linear(19, hidden_size))
        # self.obs = init_(nn.Linear(7, hidden_size))
        # self.hidden = init_(nn.Linear(hidden_size, hidden_size))
        # self.active = nn.Tanh()
        # self.attn = nn.Linear(hidden_size * 2, 2)

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        # x1 = inputs[:, :19]
        # x2 = inputs[:, 19:]
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        # a_out1 = self.target(x1)
        # a_out1 = self.active(a_out1)
        # a_out2 = self.obs(x2)
        # a_out2 = self.active(a_out2)
        # a_out = torch.cat((a_out1, a_out2), 1)  # (1, 128)
        # a_out_z = torch.cat((a_out1.unsqueeze(1), a_out2.unsqueeze(1)), 1)  # (1, 2, 64)
        # # a_hidden_out = self.hidden(a_out)
        # a_attn_weights = F.softmax(self.attn(a_out), dim=1)
        # # print(a_attn_weights.size(), a_out_z.size())
        # a_attn_applied = torch.bmm(a_attn_weights.unsqueeze(1), a_out_z)  # (8000, 1, 64)
        # a_attn_applied = a_attn_applied.squeeze(1)  # (8000, 64)
        # # a_out = a_out.unsqueeze(0)
        # # output = torch.cat((a_out, a_attn_applied), 1)
        # # output = self.attn_combine(output)
        # a_out = self.hidden(a_attn_applied)
        # # a_out = self.hidden(a_out)
        # a_out = self.active(a_out)
        # #
        # c_out1 = self.target(x1)
        # c_out1 = self.active(c_out1)
        # c_out2 = self.obs(x2)
        # c_out2 = self.active(c_out2)
        # c_out = torch.cat((c_out1, c_out2), 1)  # (8000, 128)
        # c_out_z = torch.cat((c_out1.unsqueeze(1), c_out2.unsqueeze(1)), 1)  # (8000, 2, 64)
        # # c_hidden_out = self.hidden(c_out)
        # c_attn_weights = F.softmax(self.attn(c_out), dim=1)
        # c_attn_applied = torch.bmm(c_attn_weights.unsqueeze(1), c_out_z)  # (8000, 1, 64)
        # c_attn_applied = c_attn_applied.squeeze(1)  # (8000, 64)
        # # c_attn_applied = torch.bmm(c_attn_weights.unsqueeze(0), c_out.unsqueeze(0))
        # # output = torch.cat((c_out, c_attn_applied), 1)
        # # output = self.attn_combine(output)
        # c_out = self.hidden(c_attn_applied)
        # # c_out = self.hidden(c_out)
        # c_out = self.active(c_out)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        # return self.critic_linear(c_out), a_out, rnn_hxs
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
