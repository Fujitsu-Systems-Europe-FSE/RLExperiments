from torch import nn
from model.ddpg import DDPGNet

import torch


class Critic(DDPGNet):

    def __init__(self, opts):
        super().__init__(opts)

    def _build_structure(self):
        n_layers = 2
        # TODO Batchnorm currently has detrimental effects
        batchnorm = False

        self._state_stages = nn.Sequential(
            Critic.build_stack(self._opts.n_states, 16, with_bn=batchnorm),
            Critic.build_stack(16, 32, with_bn=batchnorm)
        )

        self._action_stages = nn.Sequential(
            Critic.build_stack(self._opts.n_actions, 32, with_bn=batchnorm),
        )

        self._stages = []
        for i in range(n_layers):
            in_dim = 64 if i == 0 else self._opts.hidden_dim
            self._stages.append(Critic.build_stack(in_dim, self._opts.hidden_dim, with_bn=batchnorm))

        self._stages = nn.Sequential(
            *self._stages,
            nn.Linear(self._opts.hidden_dim, self._opts.n_actions, bias=False)
        )

    def forward(self, state, action):
        x_state = self._state_stages(state)
        x_action = self._action_stages(action)
        x = torch.cat([x_state, x_action], dim=-1)
        return self._stages(x)

    @staticmethod
    def model_name():
        return 'Critic'

    def _weight_init(self):
        pass
        # self._state_stages.append(get_initializer('relu'))
        # self._action_stages.append(get_initializer('relu'))
        # self._stages.append(get_initializer('relu'))
