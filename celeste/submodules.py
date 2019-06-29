import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def conv(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dilation=1, transpose=False, bias=False, dimension = 2,relu=True):
    if dimension == 2:
        convfn = nn.Conv2d
        convfnt = nn.ConvTranspose2d
    elif dimension == 1:
        convfn = nn.Conv1d
        convfnt = nn.ConvTranspose1d
    if not transpose:
        res = convfn(in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=((kernel_size - 1) * dilation) // 2 if dilation > 1 else pad,
                        dilation=dilation,
                        bias=bias)
    else:
        res = convfnt(in_channels=in_planes,
                                 out_channels=out_planes,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=pad,
                                 dilation=dilation,
                                 bias=bias,
                                 output_padding=1)
    if relu:
        return nn.Sequential(res, nn.ReLU())
    else:
        return res


def convbn(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dilation=1, transpose=False, bias=False, dimension = 2,relu = True):
    kwargs = {
        'kernel_size': kernel_size,
        'stride': stride,
        'pad': pad,
        'dilation': dilation,
        'transpose': transpose,
        'bias': bias,
        'dimension': dimension,
        'relu': relu,
    }
    if not relu: 
        return nn.Sequential(
            conv(in_planes, out_planes, **kwargs),
            nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(
            conv(in_planes, out_planes, **kwargs),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())
        
def convbn_resnet(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dilation=1, transpose=False, bias=False, dimension = 2,relu = True):
    kwargs = {
        'kernel_size': kernel_size,
        'stride': stride,
        'pad': pad,
        'dilation': dilation,
        'transpose': transpose,
        'bias': bias,
        'dimension': dimension,
        'relu': False
    }
    if not relu: 
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            conv(in_planes, out_planes, **kwargs))
            
    else:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            conv(in_planes, out_planes, **kwargs))
            

class ResNetModule(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 transpose=False,
                 bias=True,
                 bn=True,
                 causal=False,
                 dimension=2):
        super(ResNetModule, self).__init__()
        self.causal = kernel_size - 1 if causal else 0
        conv_fn = convbn_resnet if bn else conv
        self.convstart = conv_fn(in_planes, out_planes//4, transpose=transpose, bias=bias, dimension=dimension)
        self.convmid = conv_fn(out_planes//4,
                               out_planes//4,
                               kernel_size=kernel_size,
                               stride=stride,
                               pad=pad,
                               dilation=dilation,
                               transpose=transpose,
                               bias=bias,
                               dimension=dimension)
        self.convend = conv_fn(out_planes//4, out_planes, transpose=transpose, bias=bias, dimension=dimension)

    def forward(self, input):
        out = self.convstart(input)
        if self.causal:
          out = torch.cat((torch.zeros_like(out[:, :, :self.causal, ...]),
                           out[:, :, :-self.causal, ...]), dim=2)
        out = self.convmid(out)
        out = self.convend(out)
        return out + input