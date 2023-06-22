from torch import nn
from apheleia.model import NeuralNet


class DDPGNet(NeuralNet):

    @staticmethod
    def build_stack(in_dim, out_dim, with_bn=True):
        stage = [nn.Linear(in_dim, out_dim)]
        if with_bn:
            stage.append(nn.BatchNorm1d(out_dim))
        stage.append(nn.ReLU())
        return nn.Sequential(*stage)
