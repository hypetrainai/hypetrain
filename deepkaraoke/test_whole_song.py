# coding: utf-8
# Defining new flags needs to come first before any other imports.
import GLOBALS
GLOBALS.parser.add_argument('--idx', type=int, default=0, help='Index from test.pkl.gz to use.')
GLOBALS.parser.add_argument('--output', type=str, default='outputs/test', help='Output prefix')
FLAGS = GLOBALS.FLAGS

import dataloader
import importlib
import numpy as np
import os
import pickle as pkl
import struct
import torch
import wave

def WriteWave(data, filename):
    tensor = np.clip(data, -1.0, 1.0)
    tensor_list = [int(32767.0 * x) for x in tensor]
    wave_writer = wave.open(filename, 'wb')
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(FLAGS.sample_rate)
    tensor_enc = b''
    bytelist = [struct.pack('<h',v) for v in tensor_list]
    tensor_enc = tensor_enc.join(bytelist)
    wave_writer.writeframes(tensor_enc)
    wave_writer.close()


if __name__ == '__main__':
    NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
    model = NNModel()
    model = model.cuda()
    model_state = torch.load(os.path.join(FLAGS.log_dir, 'model_%s.pt' % FLAGS.checkpoint))
    model.load_state_dict(model_state['state_dict'])
    model.eval()

    dataset = dataloader.KaraokeDataLoader(os.path.join(FLAGS.data_dir, 'test.pkl.gz'), batch_size = 1)
    song_data = [dataset.get_single_segment(extract_idx = FLAGS.idx, start_value = 0, sample_length = 0)]
    data_in = model.preprocess(song_data)
    with torch.no_grad():
        data_out = model.predict(data_in)

    WriteWave(song_data[0].data[0], '%s_onvocal.wav' % FLAGS.output)
    WriteWave(song_data[0].data[1], '%s_offvocal.wav' % FLAGS.output)
    WriteWave(data_out, '%s_predicted.wav' % FLAGS.output)
