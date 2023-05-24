from torch import nn
from apheleia.model import NeuralNet


class DeepQNet(NeuralNet):

    def __init__(self, opts):
        super(DeepQNet, self).__init__()
        n_layers = 2
        self._stages = []
        for i in range(n_layers):
            in_dim = opts.n_observations if i == 0 else opts.hidden_dim
            self._stages.append(DeepQNet.build_stack(in_dim, opts.hidden_dim))
        self._stages = nn.Sequential(
            *self._stages,
            DeepQNet.build_stack(opts.hidden_dim, opts.n_actions, with_relu=False)
        )

    @staticmethod
    def build_stack(in_dim, out_dim, with_relu=True):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU() if with_relu else nn.Identity()
        )

    def forward(self, x):
        return self._stages(x)


class TargetNet(DeepQNet):
    @staticmethod
    def model_name():
        return 'TargetNet'

    @staticmethod
    def _weight_init(self):
        pass


class PolicyNet(DeepQNet):
    @staticmethod
    def model_name():
        return 'PolicyNet'

    @staticmethod
    def _weight_init(self):
        pass
