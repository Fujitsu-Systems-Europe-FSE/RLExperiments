from torch import nn
from apheleia.model import NeuralNet

import torch


class Critic(NeuralNet):

    def __init__(self, opts):
        super().__init__()
        n_layers = 3
        self._stages = []
        for i in range(n_layers):
            in_dim = opts.n_states + opts.n_actions if i == 0 else opts.hidden_dim
            self._stages.append(Critic.build_stack(in_dim, opts.hidden_dim))

        self._stages = nn.Sequential(
            *self._stages,
            nn.Linear(opts.hidden_dim, opts.n_actions, bias=False)
        )

    def forward(self, state, action):
        return self._stages(torch.cat([state, action], dim=-1))

    @staticmethod
    def build_stack(in_dim, out_dim, with_bn=True):
        stage = [nn.Linear(in_dim, out_dim)]
        if with_bn:
            stage.append(nn.BatchNorm1d(out_dim))
        stage.append(nn.ReLU())
        return nn.Sequential(*stage)

    @staticmethod
    def model_name():
        return 'Critic'

    @staticmethod
    def _weight_init(self):
        pass
