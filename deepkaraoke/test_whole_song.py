# coding: utf-8
# Defining new flags needs to come first before any other imports.
import GLOBALS
GLOBALS.parser.add_argument('--idx', type=int, default=0, help='Index from test to use.')
GLOBALS.parser.add_argument('--output', type=str, default='outputs/test', help='Output prefix')
FLAGS = GLOBALS.FLAGS

import dataloader
import importlib
import numpy as np
import os
import struct
import torch
import scipy.io.wavfile
import utils

def WriteWave(data, filename):
    tensor = np.clip(data, -1.0, 1.0).astype(np.float32)
    scipy.io.wavfile.write(filename, FLAGS.sample_rate, tensor)


if __name__ == '__main__':
    NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
    model = NNModel()
    model = model.cuda()
    utils.LoadModel(model)
    model.eval()

    dataset = dataloader.KaraokeDataLoader(os.path.join(FLAGS.data_dir, 'test'), batch_size = 1)
    song_data = [dataset.get_single_segment(extract_idx = FLAGS.idx, start_value = 0, sample_length = 0)]
    data_in = model.preprocess(song_data)
    with torch.no_grad():
        data_out = model.predict(data_in)

    WriteWave(song_data[0].data[0], '%s_onvocal.wav' % FLAGS.output)
    WriteWave(song_data[0].data[1], '%s_offvocal.wav' % FLAGS.output)
    WriteWave(data_out, '%s_predicted.wav' % FLAGS.output)
