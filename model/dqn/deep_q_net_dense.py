from abc import ABC
from torch import nn
from model import build_dense_stack
from apheleia.model import NeuralNet


class DeepQNet(NeuralNet, ABC):

    def __init__(self, opts):
        super(DeepQNet, self).__init__(opts)

    def _build_structure(self):
        n_layers = 2
        stages = []
        for i in range(n_layers):
            in_dim = self._opts.n_states if i == 0 else self._opts.hidden_dim
            stages.append(build_dense_stack(in_dim, self._opts.hidden_dim, with_bn=False))

        self._stages = nn.Sequential(*stages)
        self._proj = nn.Linear(self._opts.hidden_dim, self._opts.n_actions, bias=False)

    def forward(self, x):
        x = self._stages(x)
        return self._proj(x)

    @staticmethod
    def model_name():
        return 'PolicyNet'

    def _weight_init(self):
        pass
