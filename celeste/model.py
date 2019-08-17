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


class ConvModel(Model):

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


class ResNetIm2Value(ConvModel):

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

def agg_node(in_planes, out_planes):
    return nn.Sequential(
        submodules.convbn(in_planes, in_planes, kernel_size=3, stride=1, pad=1),
        submodules.convbn(in_planes, out_planes, kernel_size=3, stride=1, pad=1),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        submodules.convbn(in_planes, out_planes, kernel_size=3, stride=1, pad=1)
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )

class FPNNet(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=False):
        super(FPNNet, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)
        self.agg4 = agg_node(256, 128)
        
        # Upshuffle layers
        self.up1 = upshuffle(128,128,8)
        self.up2 = upshuffle(128,128,4)
        self.up3 = upshuffle(128,128,2)
        
        # Depth prediction
        self.predict1 = smooth(512, 128)
        self.predict2 = predict(128, 32)
        
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
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size()
        
        # Bottom-up
        c1 = self.layer0(x)
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
        _,_,H,W = d2.size()
        vol = torch.cat( [ F.upsample(d, size=(H,W), mode='bilinear') for d in [d5,d4,d3,d2] ], dim=1 )
        
        # return self.predict2( self.up4(self.predict1(vol)) )
        return self.predict2( self.predict1(vol) )     # img : depth = 4 : 1 