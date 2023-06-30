from torch import nn
from apheleia.model.losses import Loss


class DDPGLoss(Loss):

    def __init__(self, opts):
        super().__init__(opts)
        self._mse_loss = nn.MSELoss()
        self.components = ['loss/critic', 'loss/actor']

    def decompose(self):
        components = {'loss/critic': [self._critic_loss_value]}
        if hasattr(self, '_actor_loss_value') and self._actor_loss_value is not None:
            components['loss/actor'] = [self._actor_loss_value]
        return components

    def compute(self, prediction, target):
        raise NotImplemented()

    def compute_critic_loss(self, prediction, target):
        super().compute(prediction, target)
        self._critic_loss_value = self._mse_loss(prediction, target)
        return self._critic_loss_value

    def compute_actor_loss(self, critic, actions, states):
        self._actor_loss_value = -critic(states, actions).mean()
        return self._actor_loss_value
