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
                                       nn.BatchNorm1d(out_planes))
    else:
        return nn.Sequential(nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation = dilation, bias=False,output_padding=1),
                         nn.BatchNorm1d(out_planes))
        

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
nn.BatchNorm3d(out_planes))

