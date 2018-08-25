import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn_1d(in_planes, out_planes, kernel_size, stride, pad, dilation, transpose=False):
    if not transpose:
        return nn.Sequential(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, 
                                       stride=stride, padding=dilation if dilation > 1 else pad, 
                                       dilation = dilation, bias=False),
                                       nn.BatchNorm1d(out_planes)
                                       )
    else:
        return nn.Sequential(nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation = dilation, bias=False,output_padding=1),
                         nn.BatchNorm1d(out_planes)
                                      )
        

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
nn.BatchNorm3d(out_planes))

class ResNetModule1d(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation, transpose=False):
        
        super(ResNetModule1d, self).__init__()
        
        self.convstart = convbn_1d(in_planes, out_planes//4, 1, stride=1, pad=0, dilation=1, transpose=transpose)
        self.convmid = convbn_1d(out_planes//4, out_planes//4, kernel_size, stride, pad, dilation, transpose)
        self.convend = convbn_1d(out_planes//4, out_planes, 1, stride=1, pad=0, dilation=1, transpose=transpose)
        self.relustart = nn.ReLU()
        self.relumid = nn.ReLU()
        self.reluend = nn.ReLU()
        
    def forward(self, input):
        #print(input.shape)
        out = self.convstart(input)
        #print(out.shape)
        out = self.relustart(out)
        #print(out.shape)
        out = self.convmid(out)
        #print(out.shape)
        out = self.relumid(out)
        #print(out.shape)
        out = self.convend(out)
        #print(out.shape)
        
        
        return self.reluend(out+input)
        