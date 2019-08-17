from absl import flags
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import submodules
import utils

FLAGS = flags.FLAGS


class Model(nn.Module):

  def set_inputs(self, i, input_frame, extra_channels):
    raise NotImplementedError()

  def _get_inputs(self, i):
    raise NotImplementedError()


class ResNetIm2Value(Model):

  def __init__(self, in_dim, out_dim, use_softmax=True):
    super(ResNetIm2Value, self).__init__()

    self.use_softmax = use_softmax

    layer_defs = []
    layer_defs.append(submodules.convbn(in_dim, 64, kernel_size=3, pad=1, stride=2))
    layer_defs.append(submodules.convbn(64, 64, kernel_size=3, pad=1, stride=2))
    layer_defs.append(submodules.convbn(64, 64, kernel_size=3, pad=1, stride=2))

    for i in range(3):
      layer_defs.append(submodules.ResNetModule(64, 64, kernel_size=3, pad=1))
    layer_defs.append(submodules.convbn(64, 128, kernel_size=3, pad=1, stride=2))

    for i in range(3):
        layer_defs.append(submodules.ResNetModule(128, 128, kernel_size=3, pad=1))
    layer_defs.append(submodules.convbn(128, 256, kernel_size=3, pad=1, stride=2))

    for i in range(3):
      layer_defs.append(submodules.ResNetModule(256, 256, kernel_size=3, pad=1))

    layer_defs.append(submodules.convbn(256, 256, kernel_size=3, pad=1, stride=2))

    fc_input = 9 * 15 * 256
    layer_defs_linear = []
    layer_defs_linear.append(nn.Linear(fc_input, 512))
    layer_defs_linear.append(nn.ReLU())
    layer_defs_linear.append(nn.Linear(512, 256))
    layer_defs_linear.append(nn.ReLU())
    layer_defs_linear.append(nn.Linear(256, out_dim))

    self.operation_stack = nn.Sequential(*layer_defs)
    self.operation_stack_linear = nn.Sequential(*layer_defs_linear)

  def forward(self, i):
    inputs = self._get_inputs(i)
    out = self.operation_stack(inputs)
    out = out.view(inputs.shape[0], -1)
    out = self.operation_stack_linear(out)
    if self.use_softmax:
      out = F.softmax(out, 1)
    return out

  def set_inputs(self, i, input_frame, extra_channels):
    if i == 0:
      self.frame_buffer = torch.stack([input_frame] * FLAGS.context_frames, 0)
      self.extra_channels = [extra_channels]
    else:
      self.frame_buffer = torch.cat([self.frame_buffer, input_frame.unsqueeze(0)], 0)
      self.extra_channels.append(extra_channels)
    utils.assert_equal(i + FLAGS.context_frames, self.frame_buffer.shape[0])
    utils.assert_equal(i, len(self.extra_channels) - 1)

  def _get_inputs(self, i):
    input_frames = self.frame_buffer[i:i+FLAGS.context_frames]
    # [time, channels, height, width] -> [time * channels, height, width]
    input_frames = torch.reshape(input_frames, [-1, FLAGS.image_height, FLAGS.image_width])
    input_frames = torch.cat([input_frames, self.extra_channels[i]], 0)
    input_frames = input_frames.unsqueeze(0)
    return input_frames
