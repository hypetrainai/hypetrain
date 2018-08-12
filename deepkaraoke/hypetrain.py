from IPython.display import Audio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import csv, pickle, os, sys
import os
import scipy.io.wavfile as wav
import matplotlib
import matplotlib.pyplot as plt
import gzip
import pickle as pkl
from submodules import *
from collections import OrderedDict
import utils

import dataloader

batch_size = 24

train_dataset = dataloader.KaraokeDataLoader('data/train.pkl.gz', batch_size = batch_size)

test_dataset = dataloader.KaraokeDataLoader('data/test.pkl.gz', batch_size = batch_size)

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
        layer_defs.append(nn.Sequential(convbn_1d(1,32,3,1,1,1), nn.ReLU()))
        for i in range(11):
            layer_defs.append(nn.Sequential(convbn_1d(32,32,3,1,1,2**(1+i)), nn.ReLU()))
        for i in range(11):
            layer_defs.append(nn.Sequential(convbn_1d(32,32,3,1,1,2**(1+i)), nn.ReLU()))
        layer_defs.append(nn.Sequential(nn.Conv1d(32,1,3,1,1,1)))

        self.full = nn.Sequential(*layer_defs)

        #self.full = self.op_dict['layer1']

        #self.full = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, \
        #                         self.layer6, self.layer7, self.layer8, self.layer9)

    def forward(self, x):


        out = self.full(x) + x

        return out 

model = naive_generator()
model = nn.DataParallel(model).cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.1)
max_step = 1000000

try:
    for step in range(max_step):

        data = train_dataset.get_random_batch(10000)
        #data_vocal = np.array([data[i].data[0] for i in range(batch_size)], dtype=np.float32)
        #data_offvocal = np.array([data[i].data[1] for i in range(batch_size)], dtype=np.float32)
        data_vocal, data_offvocal = zip(*[d.data for d in data])

        data_vocal = torch.Tensor(data_vocal).cuda().unsqueeze(1)
        data_offvocal = torch.Tensor(data_offvocal).cuda().unsqueeze(1)

        prediction = model.forward(data_vocal)
        loss = torch.mean((prediction - data_offvocal)**2)/10000000.0

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step%100 == 0:
            print('Oh no! Your training loss is %.3f'%(loss))
            
        if step%2500 == 0:
            print('Saving model!')
            torch.save(model.state_dict(),'models/checkpoint.pt')
            print('Done!')
        
        if step%20000 == 0 and step>0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
            
except KeyboardInterrupt:
    pass
torch.cuda.empty_cache()

print('Writing predictions')
model.eval()
with torch.no_grad():

    data = train_dataset.get_random_batch(-1)[0]
    prediction = model.forward(torch.Tensor(data.data[0]).unsqueeze(0).unsqueeze(1).cuda())
    try:
        os.mkdir('out')
    except OSError:
        pass
    wav.write('out/predicted.wav', 44100, prediction.detach().cpu().numpy())
    wav.write('out/gt.wav', 44100, data.data[1])
    wav.write('out/onvocal.wav', 44100, data.data[0])
