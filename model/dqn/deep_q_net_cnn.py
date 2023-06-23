from model import feat_extractor
from model.dqn.deep_q_net_dense import DeepQNet as DeepQNetDense


class DeepQNet(DeepQNetDense):

    def _build_structure(self):
        super()._build_structure()
        self._feat_extractor = feat_extractor(self._opts.with_bn)

    def forward(self, x):
        x = self._feat_extractor(x)
        x = self._stages(x)
        return self._proj(x)
