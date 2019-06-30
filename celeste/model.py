import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import submodules
from class2button import class2button

class ResNetIm2Value(nn.Module):
    
    
    def __init__(self, FLAGS, out_dim = None):
        super(ResNetIm2Value, self).__init__()
        
        #960 x 540
        
        self.FLAGS = FLAGS
        self.num_actions = FLAGS.num_actions
        if out_dim is not None:
            self.num_actions = out_dim
        self.H = FLAGS.image_height
        self.W = FLAGS.image_width
        self.C = FLAGS.image_channels
        self.context_frames = FLAGS.context_frames
        
        fc_input = 7*13*256
        
        layer_defs = []
        layer_defs_linear = []
        layer_defs.append(submodules.convbn(self.context_frames*self.C, 64, kernel_size = 5, stride=2))
        layer_defs.append(submodules.convbn(64, 64, kernel_size = 3, stride=2))
        layer_defs.append(submodules.convbn(64, 64, kernel_size = 3, stride=2))
        
        for i in range(3):
            layer_defs.append(submodules.ResNetModule(64, 64, kernel_size=3, pad=1))
        layer_defs.append(submodules.convbn(64, 128, kernel_size=3, stride=2))
        
        for i in range(3):
            layer_defs.append(submodules.ResNetModule(128, 128, kernel_size=3, pad=1))
        layer_defs.append(submodules.convbn(128, 256, kernel_size=3, stride=2))
        
        for i in range(3):
            layer_defs.append(submodules.ResNetModule(256, 256, kernel_size=3, pad=1))
        
        layer_defs.append(submodules.convbn(256, 256, kernel_size=3, stride=2))
        
        layer_defs_linear.append(nn.Linear(fc_input, 512))
        #layer_defs_linear.append(nn.BatchNorm1d(512))
        layer_defs_linear.append(nn.ReLU())
        layer_defs_linear.append(nn.Linear(512, 256))
        #layer_defs_linear.append(nn.BatchNorm1d(256))
        layer_defs_linear.append(nn.ReLU())
        layer_defs_linear.append(nn.Linear(256, self.num_actions))
                          
        
        
        #WARNING NOT DONE - NEED TO CODE UP LINEAR LAYERS
        
        
        #layer_defs.append(submodules.conv(_CONV_CHANNELS, output_channels))
        self.operation_stack = nn.Sequential(*layer_defs)
        self.operation_stack_linear = nn.Sequential(*layer_defs_linear)
    
    def forward(self, input):
        out = self.operation_stack(input)
        out = out.view(input.shape[0],-1)
        out = self.operation_stack_linear(out)
        out = F.softmax(out,1)
        
        sample, sample_mapped = self.sample(out)
        
        return out, sample, sample_mapped
    
    def sample(self, scores):
        scores = scores.detach().cpu()
        dist = torch.distributions.categorical.Categorical(probs = scores)
        sample = dist.sample()
        sample_mapped = [class2button[int(sample[i].numpy())] for i in range(len(sample))]
        
        return sample, sample_mapped

        
        
        
