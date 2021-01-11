# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
from collections import OrderedDict
import math

import torch
import torch.nn as nn


class BRC(nn.Sequential):
    """Abbreviation of BN-ReLU-Conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dropout_rate=0.0, groups=1):
        super(BRC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        if dropout_rate > 0:
            self.add_module('drop', nn.Dropout(dropout_rate))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


class _DenseLayer(nn.Module):
    def __init__(self, n_channels, growth_rate):
        super(_DenseLayer, self).__init__()
        self.brc_1 = BRC(n_channels, 4*growth_rate, kernel_size=1)
        self.brc_2 = BRC(4*growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        x_ = self.brc_1(x)
        x_ = self.brc_2(x_)
        return torch.cat([x, x_], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, n_channels, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = _DenseLayer(n_channels + i*growth_rate, growth_rate)
            self.add_module('layer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        if in_channels != out_channels:
            self.brc = BRC(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if hasattr(self, 'brc'):
            x = self.brc(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, block_config, num_classes=10, growth_rate=12, compression=1.0):

        # Network-level hyperparameters
        self.block_config = block_config
        self.n_classes = num_classes
        self.growth_rate = growth_rate
        self.compression = compression

        assert 0 < self.compression <= 1, '0 < compression <= 1'

        super(DenseNet, self).__init__()

        i_channels = 2 * self.growth_rate
        i_features = [
            ('conv0', nn.Conv2d(3, i_channels, kernel_size=3, stride=1, padding=1, bias=False)),
        ]
        last_pool = 8
        self.features = nn.Sequential(OrderedDict(i_features))

        n_channels = i_channels
        for i, n_layers in enumerate(self.block_config):
            block = _DenseBlock(n_layers=n_layers, n_channels=n_channels, growth_rate=self.growth_rate)
            self.features.add_module('block%d' % (i + 1), block)
            n_channels = n_channels + n_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                trans = _Transition(in_channels=n_channels, out_channels=int(n_channels * self.compression))
                self.features.add_module('trans%d' % (i + 1), trans)
                n_channels = int(n_channels * self.compression)

        self.features.add_module('norm_last', nn.BatchNorm2d(n_channels))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))
        self.features.add_module('pool_last', nn.AvgPool2d(last_pool))

        self.classifier = nn.Linear(n_channels, self.n_classes)

        self.reset()

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet40(**kwargs):
    return DenseNet(block_config=[6, 6, 6], growth_rate=12, compression=1.0, **kwargs)