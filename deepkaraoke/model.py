import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from submodules import convbn_1d

class ModuleDict():
    def __init__(self):
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class naive_generator(nn.Module, ModuleDict):
    def __init__(self):
        super(naive_generator, self).__init__()

        #self.op_dict = ModuleDict(self)

        layer_defs = []
        layer_defs.append(convbn_1d(1,32,3,1,1,1))
        for i in range(11):
            layer_defs.append(convbn_1d(32,32,3,1,1,2**(1+i)))
        for i in range(11):
            layer_defs.append(convbn_1d(32,32,3,1,1,2**(1+i)))
        layer_defs.append(nn.Conv1d(32,1,3,1,1,1))

        self.full = nn.Sequential(*layer_defs)

    def forward(self, x):


        out = self.full(x)

        return out 