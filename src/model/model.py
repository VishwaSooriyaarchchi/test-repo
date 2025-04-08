# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class TargetNet(nn.Module):
    """ TargetNet for microRNA target prediction """
    def __init__(self, model_cfg, with_esa, dropout_rate):
        super(TargetNet, self).__init__()
        num_channels = model_cfg.num_channels
        num_blocks = model_cfg.num_blocks

        if not with_esa: self.in_channels, in_length = 8, 40
        else:            self.in_channels, in_length = 10, 50
        out_length = np.floor(((in_length - model_cfg.pool_size) / model_cfg.pool_size) + 1)

        self.stem = self._make_layer(model_cfg, num_channels[0], num_blocks[0], dropout_rate, stem=True)
        self.stage1 = self._make_layer(model_cfg, num_channels[1], num_blocks[1], dropout_rate, use_se=True)  # using SE here
        self.stage2 = self._make_layer(model_cfg, num_channels[2], num_blocks[2], dropout_rate, use_se=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.max_pool = nn.MaxPool1d(model_cfg.pool_size)
        self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)

    def _make_layer(self, cfg, out_channels, num_blocks, dropout_rate, stem=False, use_se=False):
        layers = []
        for b in range(num_blocks):
            if stem:
                layers.append(Conv_Layer(self.in_channels, out_channels, cfg.stem_kernel_size, dropout_rate,
                                        post_activation=b < num_blocks - 1))
            else:
                layers.append(ResNet_Block(self.in_channels, out_channels, cfg.block_kernel_size, dropout_rate,
                                            skip_connection=cfg.skip_connection, use_se=use_se))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.relu(x))
        x = self.max_pool(x)
        x = x.reshape(len(x), -1)
        x = self.linear(x)

        return x


def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
    """ kx1 convolution with padding without bias """
    layers = []
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False))
    return nn.Sequential(*layers)


class Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, post_activation):
        super(Conv_Layer, self).__init__()
        self.conv = conv_kx1(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.post_activation = post_activation

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.post_activation:
            out = self.relu(out)
            out = self.dropout(out)
        return out



class ResNet_Block(nn.Module):
    """
    ResNet Block with an optional Squeeze-and-Excitation (SE) layer.
    -- ReLU-Dropout-Conv_kx1 - ReLU-Dropout-Conv_kx1, then optional SE.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, skip_connection, use_se=False, reduction=16):
        super(ResNet_Block, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.conv1 = conv_kx1(in_channels, out_channels, kernel_size)
        self.conv2 = conv_kx1(out_channels, out_channels, kernel_size)
        self.skip_connection = skip_connection
        self.use_se = use_se
        
        # Optionally include a projection for the skip connection
        if skip_connection and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = None
        
        # Instantiate the SE block if desired
        if self.use_se:
            self.se = SEBlock(out_channels, reduction)
        else:
            self.se = None

    def forward(self, x):
        identity = x
        out = self.dropout(self.relu(x))
        out = self.conv1(out)
        out = self.dropout(self.relu(out))
        out = self.conv2(out)

        # Add skip connection
        if self.skip_connection:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out += identity

        # Apply the SE block if enabled
        if self.se is not None:
            out = self.se(out)

        return out

    
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1)
        # Scale: Multiply input by the activation
        return x * y.expand_as(x)

