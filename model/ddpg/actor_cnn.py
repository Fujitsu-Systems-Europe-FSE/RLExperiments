from model import feat_extractor
from torch.nn import functional as F
from model.ddpg.actor_dense import Actor as ActorDense


class Actor(ActorDense):

    def _build_structure(self):
        super()._build_structure()
        self._feat_extractor = feat_extractor(self._opts.with_bn)

    def forward(self, x):
        x = self._feat_extractor(x)
        x = self._stages(x)
        return F.tanh(self._proj(x))
