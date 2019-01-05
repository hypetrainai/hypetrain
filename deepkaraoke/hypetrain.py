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
import time
import scipy.io.wavfile as wav
import matplotlib
import matplotlib.pyplot as plt
import gzip
import pickle as pkl
from collections import OrderedDict
import importlib
import utils
import dataloader
from GLOBALS import FLAGS, GLOBAL

train_dataset = dataloader.KaraokeDataLoader('data/train.pkl.gz', batch_size = FLAGS.batch_size)
test_dataset = dataloader.KaraokeDataLoader('data/test.pkl.gz', batch_size = FLAGS.batch_size)

NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
model = NNModel()
current_lr = FLAGS.lr / 10
optimizer = optim.Adam(model.parameters(), lr = current_lr)

start_time = time.time()
last_step = 0
for step in range(1, 100000):

    GLOBAL.current_step = step

    data = train_dataset.get_random_batch(FLAGS.train_seqlen)
    data = model.preprocess(data)
    prediction = model.forward(data)
    loss = model.loss(prediction, data)
    GLOBAL.summary_writer.add_scalar('loss_train/total', loss, step)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if step < 100 or step%100 == 0:
        print('Oh no! Your training loss is %.3f at step %d' % (loss, step))
        GLOBAL.summary_writer.add_scalar('steps_per_s', (step - last_step) / (time.time() - start_time), step)

    if step%1000 == 0:
        print('Evaluating model!')

        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            data = test_dataset.get_random_batch(FLAGS.train_seqlen)
            data = model.preprocess(data)
            prediction = model.forward(data)
            loss = model.loss(prediction, data)
            print('Oh no! Your test loss is %.3f at step %d' % (loss, step))
            GLOBAL.summary_writer.add_scalar('loss_test/total', loss, step)

            for dataset, prefix in [(train_dataset, 'eval_train'), (test_dataset, 'eval_test')]:
                torch.cuda.empty_cache()
                data = dataset.get_random_batch(200000, batch_size=1)
                # data = [dataset.get_single_segment(extract_idx=0, start_value=3000000, sample_length=200000)]
                prediction = model.predict(model.preprocess(data), prefix)
                GLOBAL.summary_writer.add_audio(prefix + '/predicted', prediction, step, sample_rate=FLAGS.sample_rate)
                GLOBAL.summary_writer.add_audio(prefix + '/gt_onvocal', data[0].data[0], step, sample_rate=FLAGS.sample_rate)
                GLOBAL.summary_writer.add_audio(prefix + '/gt_offvocal', data[0].data[1], step, sample_rate=FLAGS.sample_rate)
        torch.cuda.empty_cache()
        model.train()

    if step%2500 == 0:
        print('Saving model!')
        model_state = {
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(model_state, FLAGS.log_dir + '/model_%d.pt' % step)
        print('Uploading Prediction!')

    if step == 100:
        current_lr = FLAGS.lr

    if step%25000 == 0:
        current_lr *= 0.2

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    GLOBAL.summary_writer.add_scalar('lr', current_lr, step)

    if step%100 == 0:
        start_time = time.time()
        last_step = step
