from torch import nn
from apheleia.model.losses import Loss


class HuberLoss(Loss):

    def __init__(self, opts):
        super().__init__(opts)
        # self._huber = nn.HuberLoss()
        self._huber = nn.SmoothL1Loss()
        self.components = ['loss/huber']
        
    def decompose(self):
        return {'loss/huber': [self._huber_value]}

    def compute(self, prediction, target):
        super().compute(prediction, target)
        self._huber_value = self._huber(prediction, target)
        return self._huber_value
