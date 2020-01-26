from absl import flags
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import submodules
import utils

from torchvision.models.resnet import resnet101

FLAGS = flags.FLAGS


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.saved_states = {}

  def reset(self):
    """Resets the model for a new episode."""
    pass

  def set_inputs(self, i, input_frame, extra_channels=None):
    """Sets the inputs for step i.

    Args:
      i: The step to set.
      input_frame: Input frames of size [batch, channel, height, width].
      extra_channels: Extra channels used for this step only.
    """
    raise NotImplementedException()

  def _get_inputs(self, i):
    """Returns the inputs to use for step i.

    This may include inputs from previous steps.
    """
    raise NotImplementedException()

  def savestate(self, index):
    """Create savestate at given index."""
    pass

  def loadstate(self, index):
    """Load the state at given index."""
    pass


class ConvModel(Model):

  def reset(self):
    # shape [time, batch, channel, height, width].
    self.frame_buffer = None
    self.extra_channels = []

  def set_inputs(self, i, input_frame, extra_channels=None):
    assert input_frame.dim() == 4
    if extra_channels is not None:
      assert extra_channels.dim() == 4
    if i == 0:
      self.frame_buffer = torch.stack([input_frame] * (FLAGS.context_frames - 1), 0)
    self.frame_buffer = torch.cat([self.frame_buffer, input_frame.unsqueeze(0)], 0)
    self.extra_channels.append(extra_channels)
    utils.assert_equal(i + FLAGS.context_frames, self.frame_buffer.shape[0])
    utils.assert_equal(i, len(self.extra_channels) - 1)

  def _get_inputs(self, i):
    input_frames = self.frame_buffer[i:i+FLAGS.context_frames]
    # [time, batch, channels, height, width] -> [batch, time * channels, height, width]
    input_frames = torch.reshape(
        input_frames.transpose(0, 1),
        [FLAGS.batch_size, -1, FLAGS.input_height, FLAGS.input_width])
    if self.extra_channels[i] is not None:
      input_frames = torch.cat([input_frames, self.extra_channels[i]], 1)
    return input_frames

  def savestate(self, index):
    state = {
        'frame_buffer': self.frame_buffer[-FLAGS.context_frames:].clone().detach()
    }
    if self.extra_channels[-1] is not None:
      state['extra_channels'] = self.extra_channels[-1].clone().detach()
    self.saved_states[index] = state

  def loadstate(self, index):
    state = self.saved_states[index]
    self.frame_buffer = state['frame_buffer'].clone()
    if 'extra_channels' in state:
      self.extra_channels = [state['extra_channels'].clone()]
    else:
      self.extra_channels = [None]


class RecurrentModel(Model):

  def reset(self):
    self.inputs = []
    self.contexts = []

  def set_inputs(self, i, input_frame, extra_channels=None):
    assert input_frame.dim() == 4
    if extra_channels is not None:
      assert extra_channels.dim() == 4
      input_frame = torch.cat([input_frame, extra_channels], 1)
    self.inputs.append(input_frame)
    utils.assert_equal(i, len(self.inputs) - 1)

  def _get_inputs(self, i):
    if not self.contexts:
      self.contexts = [self.zero_state()]
    while len(self.contexts) <= i:
      self.forward(len(self.contexts) - 1)
    return self.inputs[i], self.contexts[i]

  def savestate(self, index):
    inputs, context = self._get_inputs(len(self.inputs) - 1)
    self.saved_states[index] = (
        inputs.clone().detach(),
        [x.clone().detach() for x in context])

  def loadstate(self, index):
    inputs, context = self.saved_states[index]
    self.inputs = [inputs.clone()]
    self.contexts = [[x.clone() for x in context]]


class ResNetIm2Value(ConvModel):

  def __init__(self, frame_channels, extra_channels, out_dim, use_softmax=True):
    super(ResNetIm2Value, self).__init__()

    if not isinstance(out_dim, list):
      out_dim = [out_dim]
      use_softmax = [use_softmax]
    self.out_dim = out_dim
    self.use_softmax = use_softmax
    final_out_dim = np.sum(out_dim).astype(np.int32)

    in_dim = frame_channels * FLAGS.context_frames + extra_channels
    feat_height, feat_width = FLAGS.input_height, FLAGS.input_width

    layer_defs = []
    layer_defs.append(submodules.conv(in_dim, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    layer_defs.append(submodules.conv(64, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    layer_defs.append(submodules.conv(64, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    for i in range(3):
      layer_defs.append(submodules.ResNetModule(64, 64, kernel_size=3, pad=1))
    layer_defs.append(submodules.conv(64, 128, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    for i in range(3):
        layer_defs.append(submodules.ResNetModule(128, 128, kernel_size=3, pad=1))
    layer_defs.append(submodules.conv(128, 256, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    for i in range(3):
      layer_defs.append(submodules.ResNetModule(256, 256, kernel_size=3, pad=1))

    layer_defs.append(submodules.conv(256, 256, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    fc_input = feat_height * feat_width * 256
    layer_defs_linear = []

    linear = nn.Linear(fc_input, 512)
    submodules.init(linear, fc_input, 512, nonlinearity_after='relu')
    layer_defs_linear.append(linear)
    layer_defs_linear.append(nn.ReLU())

    linear = nn.Linear(512, 256)
    submodules.init(linear, 512, 256, nonlinearity_after='relu')
    layer_defs_linear.append(linear)
    layer_defs_linear.append(nn.ReLU())

    linear = nn.Linear(256, final_out_dim)
    submodules.init(linear, 256, final_out_dim)
    layer_defs_linear.append(linear)

    self.operation_stack = nn.Sequential(*layer_defs)
    self.operation_stack_linear = nn.Sequential(*layer_defs_linear)

  def forward(self, i):
    inputs = self._get_inputs(i)
    out = self.operation_stack(inputs)
    out = out.view(inputs.shape[0], -1)
    out = self.operation_stack_linear(out)
    current_idx = 0
    outputs = []
    for posidx, pos in enumerate(self.out_dim):
      if self.use_softmax[posidx]:
        outputs.append(utils.log_softmax(out[:, current_idx:current_idx+pos]))
      else:
        outputs.append(out[:, current_idx:current_idx+pos])
      current_idx += pos
    if len(outputs) == 1:
      outputs = outputs[0]
    return outputs


def agg_node(in_planes, out_planes):
  return nn.Sequential(
    submodules.conv(in_planes, in_planes, kernel_size=3, stride=1, pad=1),
    submodules.conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1),
  )


def upshuffle(in_planes, out_planes, upscale_factor):
  return nn.Sequential(
    submodules.conv(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, pad=1),
    nn.PixelShuffle(upscale_factor),
    nn.ReLU()
  )


class FPNNet(ConvModel):

  def __init__(self, frame_channels, extra_channels, out_dim, pretrained=True, fixed_feature_weights=False, use_softmax=True):
    super(FPNNet, self).__init__()

    in_dim = frame_channels * FLAGS.context_frames + extra_channels
    feat_height, feat_width = FLAGS.input_height, FLAGS.input_width

    resnet = resnet101(pretrained=pretrained)

    # Freeze those weights
    if fixed_feature_weights:
      for p in resnet.parameters():
        p.requires_grad = False
    self.use_softmax = use_softmax
    separate_dims = in_dim - 3
    self.layer0_sep = nn.Sequential(submodules.conv(separate_dims, 64, kernel_size=7, stride=2, pad=4),
                                    nn.MaxPool2d(2, 2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.layer1 = nn.Sequential(resnet.layer1)
    self.layer2 = nn.Sequential(resnet.layer2)
    self.layer3 = nn.Sequential(resnet.layer3)
    self.layer4 = nn.Sequential(resnet.layer4)

    # Top layer
    self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1)  # Reduce channels

    # Lateral layers
    self.latlayer1 = submodules.conv(1024, 256, kernel_size=1, stride=1)
    self.latlayer2 = submodules.conv(512, 256, kernel_size=1, stride=1)
    self.latlayer3 = submodules.conv(256, 256, kernel_size=1, stride=1)

    # Smooth layers
    self.smooth1 = submodules.conv(256, 256, kernel_size=3, stride=1, pad=1)
    self.smooth2 = submodules.conv(256, 256, kernel_size=3, stride=1, pad=1)
    self.smooth3 = submodules.conv(256, 256, kernel_size=3, stride=1, pad=1)

    # Aggregate layers
    self.agg1 = agg_node(256, 128)
    self.agg2 = agg_node(256, 128)
    self.agg3 = agg_node(256, 128)
    self.agg4 = agg_node(256, 128)

    # Upshuffle layers
    self.up1 = upshuffle(128, 128, 8)
    self.up2 = upshuffle(128, 128, 4)
    self.up3 = upshuffle(128, 128, 2)

    # Predict layers
    self.predict1 = submodules.conv(512, 128, kernel_size=3, pad=1, stride=2)
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    self.predict2 = submodules.conv(128, 128, kernel_size=3, pad=1, stride=1)
    self.predict3 = submodules.conv(128, 32, kernel_size=3, pad=1, stride=2)
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    self.predict4 = submodules.conv(32, 32, kernel_size=3, pad=1, stride=1)
    self.predict5 = submodules.conv(32, 8, kernel_size=3, pad=1, stride=1)

    fc_input = feat_height * feat_width * 8
    layer_defs_linear = []

    linear = nn.Linear(fc_input, 512)
    submodules.init(linear, fc_input, 512, nonlinearity_after='relu')
    layer_defs_linear.append(linear)
    layer_defs_linear.append(nn.ReLU())

    linear = nn.Linear(512, 256)
    submodules.init(linear, 512, 256, nonlinearity_after='relu')
    layer_defs_linear.append(linear)
    layer_defs_linear.append(nn.ReLU())

    linear = nn.Linear(256, out_dim)
    submodules.init(linear, 256, out_dim)
    layer_defs_linear.append(linear)

    self.linops = nn.Sequential(*layer_defs_linear)

  def _upsample_add(self, x, y):
    '''Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

  def forward(self, i):
    x, sep = self._get_inputs(i)

    # Bottom-up
    c1 = self.layer0(x)

    c1 += self.layer0_sep(sep)

    c2 = self.layer1(c1)
    c3 = self.layer2(c2)
    c4 = self.layer3(c3)
    c5 = self.layer4(c4)

    # Top-down
    p5 = self.toplayer(c5)
    p4 = self._upsample_add(p5, self.latlayer1(c4))
    p4 = self.smooth1(p4)
    p3 = self._upsample_add(p4, self.latlayer2(c3))
    p3 = self.smooth2(p3)
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    p2 = self.smooth3(p2)

    # Top-down predict and refine
    d5, d4, d3, d2 = self.up1(self.agg1(p5)), self.up2(self.agg2(p4)), self.up3(self.agg3(p3)), self.agg4(p2)
    _, _, H, W = d2.size()
    vol = torch.cat([F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False) for d in [d5,d4,d3,d2]], dim=1)
    vol = self.predict5(self.predict4(self.predict3(self.predict2(self.predict1(vol))))).view(vol.shape[0],-1)
    out = self.linops(vol)

    if self.use_softmax:
      out = utils.log_softmax(out)

    return out

  def _get_inputs(self, i):
      input_frames = self.frame_buffer[i:i+FLAGS.context_frames]
      # [time, channels, height, width] -> [time * channels, height, width]
      input_frames = torch.reshape(input_frames, [-1, FLAGS.input_height, FLAGS.input_width])
      current_frame = input_frames[-4:-1]
      other = torch.cat([input_frames[:-4], input_frames[-1:], self.extra_channels[i]])
      return current_frame.unsqueeze(0), other.unsqueeze(0)


class SimpleLSTMModel(RecurrentModel):

  def __init__(self, frame_channels, extra_channels, out_dim, use_softmax=True):
    super(SimpleLSTMModel, self).__init__()

    self.hidden_dim = 512
    self.lstm_layers = 2
    self.use_softmax = use_softmax

    in_dim = frame_channels + extra_channels
    feat_height, feat_width = FLAGS.input_height, FLAGS.input_width

    conv_stack = []
    conv_stack.append(submodules.conv(in_dim, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    conv_stack.append(submodules.conv(64, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    conv_stack.append(submodules.conv(64, 64, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    for i in range(3):
      conv_stack.append(submodules.ResNetModule(64, 64, kernel_size=3, pad=1))
    conv_stack.append(submodules.conv(64, 128, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    for i in range(3):
        conv_stack.append(submodules.ResNetModule(128, 128, kernel_size=3, pad=1))
    conv_stack.append(submodules.conv(128, 128, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)
    conv_stack.append(submodules.conv(128, 128, kernel_size=3, pad=1, stride=2))
    feat_height, feat_width = math.ceil(feat_height / 2), math.ceil(feat_width / 2)

    fc_input = feat_height * feat_width * 128
    self.conv_stack = nn.Sequential(*conv_stack)
    self.conv_proj = nn.Linear(fc_input, self.hidden_dim)
    submodules.init(self.conv_proj, fc_input, self.hidden_dim)
    self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.lstm_layers)
    self.out_proj = nn.Linear(self.hidden_dim, out_dim)
    submodules.init(self.out_proj, self.hidden_dim, out_dim)

  def zero_state(self):
    h0 = torch.zeros(self.lstm_layers, 1, self.hidden_dim)
    c0 = torch.zeros(self.lstm_layers, 1, self.hidden_dim)
    if FLAGS.use_cuda:
      h0 = h0.cuda()
      c0 = c0.cuda()
    return h0, c0

  def forward(self, i):
    inputs, context = self._get_inputs(i)
    out = self.conv_stack(inputs)
    out = self.conv_proj(out.view(inputs.shape[0], -1))
    out, new_context = self.rnn(out.unsqueeze(0), context)
    if i == len(self.contexts) - 1:
      self.contexts.append(new_context)
    out = self.out_proj(out.squeeze(0))
    if self.use_softmax:
      out = utils.log_softmax(out)
    return out
