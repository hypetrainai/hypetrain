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
from collections import OrderedDict
import importlib
import utils
import dataloader


batch_size = 24

train_dataset = dataloader.KaraokeDataLoader('data/train.pkl.gz', batch_size = batch_size)

test_dataset = dataloader.KaraokeDataLoader('data/test.pkl.gz', batch_size = batch_size)

module_name, model_name = sys.argv[1].rsplit('.', 1)
NNModel = getattr(importlib.import_module(module_name), model_name)
model = NNModel()
optimizer = optim.Adam(model.parameters(), lr = 0.04)
max_step = 100000

try:
    for step in range(max_step):

        data = train_dataset.get_random_batch(10000)
        data = model.preprocess(data)
        prediction = model.forward(data)
        loss = model.loss(prediction, data)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step%100 == 0:
            print('Oh no! Your training loss is %.3f at step %d'%(loss, step))

        if step%2500 == 0:
            print('Saving model!')
            torch.save(model.state_dict(),'models/checkpoint.pt')
            print('Done!')

        if step%25000 == 0 and step>0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

except KeyboardInterrupt:
    pass
torch.cuda.empty_cache()

print('Writing predictions')
model.eval()

'''
with torch.no_grad():

    data = train_dataset.get_random_batch(-1)[0]
    prediction = model.predict(torch.Tensor(data.data[0]).unsqueeze(0).unsqueeze(1).cuda())
    try:
        os.mkdir('out')
    except OSError:
        pass
    wav.write('out/predicted.wav', 44100, prediction.detach().cpu().numpy())
    wav.write('out/gt.wav', 44100, data.data[1])
    wav.write('out/onvocal.wav', 44100, data.data[0])
'''
