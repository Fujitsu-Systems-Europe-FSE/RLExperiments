from torch import nn


def feat_extractor(with_bn, feats_dim=512):
    return nn.Sequential(
        build_cnn_stack(3, 32, 8, 4, 0, with_bn=with_bn),
        build_cnn_stack(32, 64, 4, 2, 0, with_bn=with_bn),
        build_cnn_stack(64, 64, 3, 1, 0, with_bn=with_bn),
        nn.Flatten(),
        nn.Linear(59904, feats_dim)
    )


def build_cnn_stack(in_channels, out_channels, kernel, stride, padding, dilation=1, with_bn=True):
    with_bias = False if with_bn else True
    # padding = (kernel - 1) // 2
    stage = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation,
                       bias=with_bias)]
    if with_bn:
        stage.append(nn.BatchNorm2d(out_channels))
    stage.append(nn.ReLU())
    return nn.Sequential(*stage)


def build_dense_stack(in_dim, out_dim, with_bn=True):
    with_bias = False if with_bn else True
    stage = [nn.Linear(in_dim, out_dim, bias=with_bias)]
    if with_bn:
        stage.append(nn.BatchNorm1d(out_dim))
    stage.append(nn.ReLU())
    return nn.Sequential(*stage)
