from abc import ABC
from torch import nn
from apheleia.model import NeuralNet


class DeepQNet(NeuralNet, ABC):

    def __init__(self, opts):
        super(DeepQNet, self).__init__()
        n_layers = 2
        self._stages = []
        for i in range(n_layers):
            in_dim = opts.n_states if i == 0 else opts.hidden_dim
            self._stages.append(DeepQNet.build_stack(in_dim, opts.hidden_dim))
        self._stages = nn.Sequential(
            *self._stages,
            nn.Linear(opts.hidden_dim, opts.n_actions, bias=False)
        )

    @staticmethod
    def build_stack(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self._stages(x)

    @staticmethod
    def model_name():
        return 'PolicyNet'

    def _weight_init(self):
        pass
