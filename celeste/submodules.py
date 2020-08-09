import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
import numpy as np


def init(m, fan_in=1, fan_out=1, nonlinearity_after='linear',
         last_conv_of_resnet_block_with_size=0, scale=None):
  """https://arxiv.org/pdf/1906.02341.pdf"""
  if not scale:
    if last_conv_of_resnet_block_with_size:
      gamma = math.sqrt(1.0 / last_conv_of_resnet_block_with_size)
    else:
      gamma = nn.init.calculate_gain(nonlinearity_after)
    scale = gamma * math.sqrt(fan_in / fan_out)
  nn.init.orthogonal_(m.weight, gain=scale)
  nn.init.constant_(m.bias, 0)


def conv(in_planes, out_planes, kernel_size=1, stride=1, pad=0, dimension=2,
         dilation=1, transpose=False, bias=True, relu=True):
  if dimension == 1:
    convfn = nn.Conv1d
    convfnt = nn.ConvTranspose1d
  elif dimension == 2:
    convfn = nn.Conv2d
    convfnt = nn.ConvTranspose2d
  else:
    raise ValueError('Invalid dimension %d' % dimension)
  if not transpose:
    res = convfn(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=((kernel_size - 1) * dilation) // 2 if dilation > 1 else pad,
        dilation=dilation,
        bias=bias)
  else:
    res = convfnt(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        dilation=dilation,
        bias=bias,
        output_padding=1)
  init(res, in_planes, out_planes, nonlinearity_after='relu' if relu else 'linear')
  if relu:
    res = nn.Sequential(res, nn.ReLU())
  return res


class ResNetModule(nn.Module):

  def __init__(self,
               in_planes,
               out_planes,
               kernel_size=1,
               stride=1,
               pad=0,
               dimension=2,
               dilation=1,
               transpose=False,
               causal=False):
    super().__init__()
    self.causal = kernel_size - 1 if causal else 0
    self.dimension = dimension
    bottleneck = (out_planes + 3) // 4
    self.convstart = conv(in_planes, bottleneck, dimension=dimension, transpose=transpose)
    init(self.convstart[0], in_planes, bottleneck, nonlinearity_after='relu')
    self.convmid = conv(bottleneck,
                        bottleneck,
                        kernel_size=kernel_size,
                        stride=stride,
                        pad=pad,
                        dimension=dimension,
                        dilation=dilation,
                        transpose=transpose)
    init(self.convmid[0], bottleneck, bottleneck, nonlinearity_after='relu')
    self.convend = conv(bottleneck, out_planes, dimension=dimension, transpose=transpose, relu=False)
    init(self.convend, bottleneck, out_planes, last_conv_of_resnet_block_with_size=3)

  def forward(self, input):
    out = F.relu(input)
    out = self.convstart(out)
    if self.causal:
      assert self.dimension == 2
      out = torch.cat((torch.zeros_like(out[:, :, :self.causal, ...]),
                       out[:, :, :-self.causal, ...]), dim=2)
    out = self.convmid(out)
    out = self.convend(out)
    return out + input
