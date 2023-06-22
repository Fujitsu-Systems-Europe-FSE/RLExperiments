from torch import nn
from model.ddpg import DDPGNet


class Actor(DDPGNet):

    def __init__(self, opts):
        super().__init__(opts)
        self._rescale_factor = self._opts.max_action

    def _build_structure(self):
        n_layers = 2
        # TODO Batchnorm currently has detrimental effects
        batchnorm = False

        self._stages = []
        for i in range(n_layers):
            in_dim = self._opts.n_states if i == 0 else self._opts.hidden_dim
            self._stages.append(Actor.build_stack(in_dim, self._opts.hidden_dim, with_bn=batchnorm))

        self._stages = nn.Sequential(*self._stages)
        self._proj = nn.Sequential(
            nn.Linear(self._opts.hidden_dim, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self._stages(x)
        return self._proj(x) * self._rescale_factor

    @staticmethod
    def model_name():
        return 'Actor'

    def _weight_init(self):
        pass
        # self._stages.apply(get_initializer('relu'))
        # self._proj.apply(get_initializer('tanh'))
