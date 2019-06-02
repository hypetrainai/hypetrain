import torch
import torch.nn as nn
import numpy as np
import submodules
import utils

class PolicyPredictor(nn.Module):
    
    super(PolicyPredictor, self).__init__()
    
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.num_actions = FLAGS.num_actions
        self.H = FLAGS.image_height
        self.W = FLAGS.image_width
        self.C = FLAGS.image_channels
        
        fc_input = self.H*self.W//8//8 * 256 * 256
        
        layer_defs = []
        layer_defs.append(submodules.convbn(input_channels, 64, kernel_size = 5, stride=2))
        
        for i in range(4):
            layer_defs.append(submodules.ResNetModule(64, 64, kernel_size=3, pad=1))
        layer_defs.append(submodules.convbn(64, 128, kernel_size=3, stride=2))
        
        for i in range(4):
            layer_defs.append(submodules.ResNetModule(128, 128, kernel_size=3, pad=1))
        layer_defs.append(submodules.convbn(128, 256, kernel_size=3, stride=2))
        
        for i in range(4):
            layer_defs.append(submodules.ResNetModule(256, 256, kernel_size=3, pad=1))
        
        #WARNING NOT DONE - NEED TO CODE UP LINEAR LAYERS
        
        
        layer_defs.append(submodules.conv(_CONV_CHANNELS, output_channels))
return nn.Sequential(*layer_defs)
        
        
