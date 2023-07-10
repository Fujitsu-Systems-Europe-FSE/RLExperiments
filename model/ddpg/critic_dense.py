from torch import nn
from torch.nn import init
from model import build_dense_stack
from apheleia.model import NeuralNet

import torch


class Critic(NeuralNet):

    def _build_structure(self):
        n_layers = 2

        self._state_stages = nn.Sequential(
            build_dense_stack(self._opts.n_states, 16, with_bn=self._opts.with_bn),
            build_dense_stack(16, 32, with_bn=self._opts.with_bn)
        )

        self._action_stages = build_dense_stack(self._opts.n_actions, 32, with_bn=self._opts.with_bn)

        stages = []
        for i in range(n_layers):
            in_dim = 64 if i == 0 else self._opts.hidden_dim
            self._stages.append(build_dense_stack(in_dim, self._opts.hidden_dim, with_bn=self._opts.with_bn))

        self._stages = nn.Sequential(*stages)
        self._proj = nn.Linear(self._opts.hidden_dim, 1, bias=False)

    def forward(self, state, action):
        x_state = self._state_stages(state)
        x_action = self._action_stages(action)
        x = torch.cat([x_state, x_action], dim=-1)
        x = self._stages(x)
        return self._proj(x)

    @staticmethod
    def model_name():
        return 'Critic'

    def _weight_init(self):
        init.uniform_(self._proj.weight.data, -3e-3, 3e-3)
        if self._proj.bias is not None:
            init.uniform_(self._proj.bias.data, -3e-3, 3e-3)
