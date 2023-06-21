from torch import nn
from apheleia.model import NeuralNet


class Actor(NeuralNet):

    def __init__(self, opts):
        super().__init__()
        n_layers = 3
        self._stages = []
        for i in range(n_layers):
            in_dim = opts.n_states if i == 0 else opts.hidden_dim
            self._stages.append(Actor.build_stack(in_dim, opts.hidden_dim))

        self._stages = nn.Sequential(
            *self._stages,
            nn.Linear(opts.hidden_dim, 1, bias=False)
            # TODO Tanh here ?
        )

    def forward(self, x):
        return self._stages(x)

    @staticmethod
    def build_stack(in_dim, out_dim, with_bn=True):
        stage = [nn.Linear(in_dim, out_dim)]
        if with_bn:
            stage.append(nn.BatchNorm1d(out_dim))
        stage.append(nn.ReLU())
        return nn.Sequential(*stage)

    @staticmethod
    def model_name():
        return 'Actor'

    @staticmethod
    def _weight_init(self):
        pass
