from torch import nn
from torch.nn import init
from model import build_dense_stack
from torch.nn import functional as F
from apheleia.model import NeuralNet


class Actor(NeuralNet):

    def __init__(self, opts):
        super().__init__(opts)
        self._rescale_factor = self._opts.max_action if hasattr(self._opts, 'max_action') else 1

    def _build_structure(self):
        n_layers = 2

        stages = []
        for i in range(n_layers):
            in_dim = self._opts.n_states if i == 0 else self._opts.hidden_dim
            stages.append(build_dense_stack(in_dim, self._opts.hidden_dim, with_bn=self._opts.with_bn))

        self._stages = nn.Sequential(*stages)
        self._proj = nn.Linear(self._opts.hidden_dim, self._opts.n_actions, bias=False)

    def forward(self, x):
        x = self._stages(x)
        return F.tanh(self._proj(x)) * self._rescale_factor

    @staticmethod
    def model_name():
        return 'Actor'

    def _weight_init(self):
        init.uniform_(self._proj.weight.data, -3e-3, 3e-3)
        if self._proj.bias is not None:
            init.uniform_(self._proj.bias.data, -3e-3, 3e-3)
