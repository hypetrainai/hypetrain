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
from fftmodel import Discriminator as disc
from tensorboardX import SummaryWriter

batch_size = 64
train_dataset = dataloader.KaraokeDataLoader(
    os.path.join(FLAGS.data_dir, 'train.pkl.gz'),
    batch_size = batch_size)
test_dataset = dataloader.KaraokeDataLoader(
    os.path.join(FLAGS.data_dir, 'test.pkl.gz'),
    batch_size = batch_size)

NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
model = NNModel()
model_D = disc()
optimizer = optim.Adam(model.parameters(), lr = FLAGS.lr)
optimizer_disc = optim.Adam(model_D.parameters(), lr = 0.0025*FLAGS.lr)
start_step = 0
if FLAGS.resume:
    start_step = utils.LoadModel(model, optimizer)
    utils.LoadModel(model_D, optimizer_disc, 'model_disc')
assert len(optimizer.param_groups) == 1
assert len(optimizer_disc.param_groups) == 1
aux_weights = None

def get_aux_weights(step, num_aux = 5, step_between = 1000):
    final = torch.zeros([num_aux]).float()
    if step < step_between:
        final[0] = 1.0
    else:
        number = step//step_between
        if number >= num_aux-1:
            final[num_aux-1] = 1.0
        else:
            percentage = float(step - number*step_between)/step_between
            final[number] = percentage
            final[number-1] = 1-percentage
    return final.cuda()

start_time = time.time()
for step in range(start_step + 1, FLAGS.max_steps):
    data = train_dataset.get_random_batch(20000)
    data = model.preprocess(data)

    GLOBAL.current_step = step

    prediction = model.forward(data)
    aux_weights = get_aux_weights(step, num_aux = len(prediction))

    if FLAGS.model_name == 'GeneratorDeepSupervision':
        aux_weights = get_aux_weights(step, num_aux = len(prediction))
        prediction = torch.sum(torch.cat([(aux_weights[i]*prediction[i]).unsqueeze(0) for i in range(len(prediction))],0),0)
    loss = model.loss(prediction, data)
    GLOBAL.summary_writer.add_scalar('loss_train/total', loss, step)

    GAN_pred = model_D.forward(prediction)
    GAN_gt = model_D.forward(data)
    #gt_disc = torch.cat([torch.zeros([GAN_pred.size(0)]).type(torch.int), torch.ones([GAN_gt.size(0)]).type(torch.int)])

    scores_pred = torch.clamp(F.softmax(GAN_pred, 1), 0.00001, 0.99999)
    scores_gt = torch.clamp(F.softmax(GAN_gt, 1), 0.00001, 0.99999)

    #print(scores_pred)

    GAN_loss = -1.0*torch.mean(torch.log(1.0-scores_pred[:,0]))
    #print(GAN_loss)
    GLOBAL.summary_writer.add_scalar('loss_train/GAN_gen_loss', GAN_loss, step)

    total_loss = loss + 0.002*GAN_loss
    total_loss.backward()

    if FLAGS.clip_grad_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    optimizer_disc.zero_grad()

    disc_iter = 3

    for disc_step in range(disc_iter):
        data = train_dataset.get_random_batch(20000)
        data = model.preprocess(data)
        prediction = model.forward(data)
        if FLAGS.model_name == 'GeneratorDeepSupervision':
            prediction = torch.sum(torch.cat([(aux_weights[i]*prediction[i]).unsqueeze(0) for i in range(len(prediction))],0),0)
        #GLOBAL.summary_writer.add_scalar('loss_train/total', loss, step)

        GAN_pred = model_D.forward(prediction)
        GAN_gt = model_D.forward(data)

        scores_pred = torch.clamp(F.softmax(GAN_pred, 1), 0.00001, 0.99999)
        scores_gt = torch.clamp(F.softmax(GAN_gt, 1), 0.00001, 0.99999)

        GAN_loss_disc = torch.mean(torch.log(1.0-scores_pred[:,0]))+torch.mean(torch.log(scores_gt[:,0]))
        GAN_loss_disc.backward()

        if FLAGS.clip_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(model_D.parameters(), FLAGS.clip_grad_norm)
        optimizer_disc.step()
        optimizer_disc.zero_grad()
        optimizer.zero_grad()
    GLOBAL.summary_writer.add_scalar('loss_train/disc_loss', GAN_loss_disc, step)

    GLOBAL.summary_writer.add_scalar('lr/gen', optimizer.param_groups[0]['lr'], step)
    GLOBAL.summary_writer.add_scalar('lr/disc', optimizer_disc.param_groups[0]['lr'], step)

    if step%25 == 0:
        print('Oh no! Your training loss is %.3f at step %d' % (loss, step))
        GLOBAL.summary_writer.add_scalar('steps/s', 25.0 / (time.time() - start_time), step)

    if step%1000 == 0 or step==1:
        print('Evaluating model!')

        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            data = test_dataset.get_random_batch(20000)
            data = model.preprocess(data)
            prediction = model.forward(data)
            if FLAGS.model_name == 'GeneratorDeepSupervision':
                prediction = torch.sum(torch.cat([(aux_weights[i]*prediction[i]).unsqueeze(0) for i in range(len(prediction))],0),0)
            loss = model.loss(prediction, data)
            print('Oh no! Your test loss is %.3f at step %d' % (loss, step))
            GLOBAL.summary_writer.add_scalar('loss_test/total', loss, step)

            for dataset, prefix in [(train_dataset, 'eval_train'), (test_dataset, 'eval_test')]:
                torch.cuda.empty_cache()
                # data = dataset.get_random_batch(500000, batch_size=1)
                data = [dataset.get_single_segment(extract_idx=0, start_value=500000, sample_length=200000)]

                prediction = model.predict(model.preprocess(data), aux_weights = aux_weights)
                #if FLAGS.model_name == 'GeneratorDeepSupervision':
                #    prediction = torch.sum(torch.cat([(aux_weights[i]*prediction[i]).unsqueeze(0) for i in range(len(prediction))],0),0)
                GLOBAL.summary_writer.add_audio(prefix + '/predicted', prediction, step, sample_rate=FLAGS.sample_rate)
                GLOBAL.summary_writer.add_audio(prefix + '/gt_onvocal', data[0].data[0], step, sample_rate=FLAGS.sample_rate)
                GLOBAL.summary_writer.add_audio(prefix + '/gt_offvocal', data[0].data[1], step, sample_rate=FLAGS.sample_rate)
                GLOBAL.summary_writer.add_audio(prefix + '/gt_vocal_diff',
                                                data[0].data[0] - data[0].data[1],
                                                step,
                                                sample_rate=FLAGS.sample_rate)
        torch.cuda.empty_cache()
        model.train()

    if step%2000 == 0:
        utils.SaveModel(step, model, optimizer)
        utils.SaveModel(step, model_D, optimizer_disc, 'model_disc')

    if step%10000 == 0:
        optimizer.param_groups[0]['lr'] *= 0.2
        optimizer_disc.param_groups[0]['lr'] *= 0.2

    if step%100 == 0:
        start_time = time.time()
