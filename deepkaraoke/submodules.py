import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def conv_1d(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dilation=1, transpose=False, bias=False):
    if not transpose:
        res = nn.Conv1d(in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=((kernel_size - 1) * dilation) // 2 if dilation > 1 else pad,
                        dilation=dilation,
                        bias=bias)
    else:
        res = nn.ConvTranspose1d(in_channels=in_planes,
                                 out_channels=out_planes,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=pad,
                                 dilation=dilation,
                                 bias=bias,
                                 output_padding=1)
    return res


def convbn_1d(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dilation=1, transpose=False, bias=False):
    kwargs = {
        'kernel_size': kernel_size,
        'stride': stride,
        'pad': pad,
        'dilation': dilation,
        'transpose': transpose,
        'bias': bias,
    }
    return nn.Sequential(
        conv_1d(in_planes, out_planes, **kwargs),
        nn.BatchNorm1d(out_planes))


class ResNetModule1d(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 transpose=False,
                 bias=False,
                 bn=True,
                 causal=False):
        super(ResNetModule1d, self).__init__()
        self.causal = kernel_size - 1 if causal else 0
        conv_fn = convbn_1d if bn else conv_1d
        self.convstart = conv_fn(in_planes, out_planes//4, transpose=transpose, bias=bias)
        self.convmid = conv_fn(out_planes//4,
                               out_planes//4,
                               kernel_size=kernel_size,
                               stride=stride,
                               pad=pad,
                               dilation=dilation,
                               transpose=transpose,
                               bias=bias)
        self.convend = conv_fn(out_planes//4, out_planes, transpose=transpose, bias=bias)

    def forward(self, input):
        out = self.convstart(input)
        out = F.relu(out)
        if self.causal:
          out = torch.cat((torch.zeros_like(out[:, :, :self.causal, ...]),
                           out[:, :, :-self.causal, ...]), dim=2)
        out = self.convmid(out)
        out = F.relu(out)
        out = self.convend(out)
        return F.relu(out + input)
