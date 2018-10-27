import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from submodules import convbn_1d
from network import Network


class naive_generator(Network):

    def BuildModel(self):
        layer_defs = []
        layer_defs.append(convbn_1d(1,32,3,1,1,1))
        for i in range(11):
            layer_defs.append(convbn_1d(32,32,3,1,1,2**(1+i)))
        for i in range(11):
            layer_defs.append(convbn_1d(32,32,3,1,1,2**(1+i)))
        layer_defs.append(nn.Conv1d(32,1,3,1,1,1))
        return nn.Sequential(*layer_defs)

    def forward(self, data):
        data_vocal = [d.data[0] for d in data]
        data_vocal = torch.Tensor(data_vocal).cuda().unsqueeze(1)
        out = self.model.forward(data_vocal)
        return out

    def loss(self, prediction, data):
        data_offvocal = [d.data[1] for d in data]
        data_offvocal = torch.Tensor(data_offvocal).cuda().unsqueeze(1)
        return torch.mean((prediction - data_offvocal)**2)/10000000.0
