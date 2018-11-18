import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def conv_1d(in_planes, out_planes, kernel_size, stride, pad, dilation, transpose=False, bias=False, wn=False):
    if not transpose:
        res = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, padding=dilation if dilation > 1 else pad,
                        dilation=dilation, bias=bias)
    else:
        res = nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size,
                                 stride=stride, padding=pad, dilation=dilation, bias=bias,
                                 output_padding=1)
    if wn:
        res = nn.utils.weight_norm(res)
    return res


def convbn_1d(in_planes, out_planes, kernel_size, stride, pad, dilation, transpose=False, bias=False, wn=False):
    return nn.Sequential(conv_1d(in_planes, out_planes, kernel_size, stride, pad, dilation, transpose, wn),
                         nn.BatchNorm1d(out_planes))


class ResNetModule1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation, transpose=False, causal=False, bias=False, bn=True, wn=False):

        super(ResNetModule1d, self).__init__()
        self.causal = kernel_size - 1 if causal else 0
        conv_fn = convbn_1d if bn else conv_1d
        self.convstart = conv_fn(in_planes, out_planes//4, 1, stride=1, pad=0, dilation=1, transpose=transpose, bias=bias, wn=wn)
        self.convmid = conv_fn(out_planes//4, out_planes//4, kernel_size, stride, pad, dilation, transpose, bias=bias, wn=wn)
        self.convend = conv_fn(out_planes//4, out_planes, 1, stride=1, pad=0, dilation=1, transpose=transpose, bias=bias, wn=wn)

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
