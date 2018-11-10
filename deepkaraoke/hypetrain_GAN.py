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
from CONSOLE_ARGS import ARGS as FLAGS
from fftmodel import Discriminator as disc
from tensorboardX import SummaryWriter

os.makedirs(FLAGS.log_dir, exist_ok=True)
writer = SummaryWriter(FLAGS.log_dir)

batch_size = 64
train_dataset = dataloader.KaraokeDataLoader('data/train.pkl.gz', batch_size = batch_size)
test_dataset = dataloader.KaraokeDataLoader('data/test.pkl.gz', batch_size = batch_size)

NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
model_D = disc(writer)
model = NNModel(writer)
optimizer = optim.Adam(model.parameters(), lr = FLAGS.lr)
optimizer_disc = optim.Adam(model_D.parameters(), lr = 0.0025*FLAGS.lr)


model_state = torch.load('trained_models/GANwithphase2/model_20000.pt')
disc_state = torch.load('trained_models/GANwithphase2/model_disc_20000.pt')

model.load_state_dict(model_state['state_dict'])
optimizer.load_state_dict(model_state['optimizer'])

model_D.load_state_dict(disc_state['state_dict'])
optimizer_disc.load_state_dict(disc_state['optimizer'])

start_time = time.time()
for step in range(1, 100000):
    data = train_dataset.get_random_batch(20000)
    data = model.preprocess(data)

    model.current_step = step


    prediction = model.forward(data)
    loss = model.loss(prediction, data)
    writer.add_scalar('loss_train/total', loss, step)

    GAN_pred = model_D.forward(prediction)
    GAN_gt = model_D.forward(data)
    #gt_disc = torch.cat([torch.zeros([GAN_pred.size(0)]).type(torch.int), torch.ones([GAN_gt.size(0)]).type(torch.int)])

    scores_pred = torch.clamp(F.softmax(GAN_pred, 1), 0.00001, 0.99999)
    scores_gt = torch.clamp(F.softmax(GAN_gt, 1), 0.00001, 0.99999)

    #print(scores_pred)

    GAN_loss = -1.0*torch.mean(torch.log(1.0-scores_pred[:,0]))
    #print(GAN_loss)
    writer.add_scalar('loss_train/GAN_gen_loss', GAN_loss, step)

    total_loss = loss + 0.002*GAN_loss

    total_loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    optimizer_disc.zero_grad()

    disc_iter = 3

    for disc_step in range(disc_iter):
        data = train_dataset.get_random_batch(20000)
        data = model.preprocess(data)
        prediction = model.forward(data)
        #writer.add_scalar('loss_train/total', loss, step)

        GAN_pred = model_D.forward(prediction)
        GAN_gt = model_D.forward(data)

        scores_pred = torch.clamp(F.softmax(GAN_pred, 1), 0.00001, 0.99999)
        scores_gt = torch.clamp(F.softmax(GAN_gt, 1), 0.00001, 0.99999)

        GAN_loss_disc = torch.mean(torch.log(1.0-scores_pred[:,0]))+torch.mean(torch.log(scores_gt[:,0]))

        GAN_loss_disc.backward()
        optimizer_disc.step()
        optimizer_disc.zero_grad()
        optimizer.zero_grad()
    writer.add_scalar('loss_train/disc_loss', GAN_loss_disc, step)

    if step%25 == 0:
        print('Oh no! Your training loss is %.3f at step %d' % (loss, step))
        writer.add_scalar('steps/s', 25.0 / (time.time() - start_time), step)

    if step%1000 == 0:
        print('Evaluating model!')

        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            data = test_dataset.get_random_batch(20000)
            data = model.preprocess(data)
            prediction = model.forward(data)
            loss = model.loss(prediction, data)
            print('Oh no! Your test loss is %.3f at step %d' % (loss, step))
            writer.add_scalar('loss_test/total', loss, step)

            for dataset, prefix in [(train_dataset, 'eval_train'), (test_dataset, 'eval_test')]:
                torch.cuda.empty_cache()
                # data = dataset.get_random_batch(500000, batch_size=1)
                data = [dataset.get_single_segment(extract_idx=0, start_value=2000000, sample_length=200000)]
                prediction = model.predict(model.preprocess(data))
                writer.add_audio(prefix + '/predicted', prediction, step, sample_rate=FLAGS.sample_rate)
                on_vocal, off_vocal = utils.Convert16BitToFloat(data[0].data[0], data[0].data[1])
                writer.add_audio(prefix + '/gt_onvocal', on_vocal, step, sample_rate=FLAGS.sample_rate)
                writer.add_audio(prefix + '/gt_offvocal', off_vocal, step, sample_rate=FLAGS.sample_rate)
        torch.cuda.empty_cache()
        model.train()

    if step%2000 == 0:
        print('Saving model!')
        model_state = {
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        disc_model_state = {
            'step': step,
            'state_dict': model_D.state_dict(),
            'optimizer': optimizer_disc.state_dict()
        }
        torch.save(model_state, FLAGS.log_dir + '/model_%d.pt' % step)
        torch.save(disc_model_state, FLAGS.log_dir + '/model_disc_%d.pt' % step)

    if step%10000 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2

    if step%100 == 0:
        start_time = time.time()
