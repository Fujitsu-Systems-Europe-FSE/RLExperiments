from model import feat_extractor
from model.ddpg.critic_dense import Critic as CriticDense

import torch


class Critic(CriticDense):

    def _build_structure(self):
        super()._build_structure()
        self._feat_extractor = feat_extractor(self._opts.with_bn)

    def forward(self, state, action):
        feats = self._feat_extractor(state)
        x_state = self._state_stages(feats)
        x_action = self._action_stages(action)
        x = torch.cat([x_state, x_action], dim=-1)
        x = self._stages(x)
        return self._proj(x)
