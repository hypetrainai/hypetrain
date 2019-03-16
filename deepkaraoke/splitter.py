import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import submodules
from network import Network
import utils
from GLOBALS import FLAGS, GLOBAL


_EMB_REDUCTION_FACTOR = 8
_EMB_DIM = 1024
_CONV_CHANNELS = 256
_CONV_LAYERS = 24
_CONV_DILATION_CYCLE = 8


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self._layers = nn.ModuleList()
        self._layers.append(submodules.conv_1d(_EMB_REDUCTION_FACTOR, _CONV_CHANNELS))
        for i in range(_CONV_LAYERS):
            self._layers.append(submodules.ResNetModule1d(_CONV_CHANNELS, _CONV_CHANNELS, kernel_size=3, pad=1, dilation=2**(i % _CONV_DILATION_CYCLE)))
        self._output = submodules.conv_1d(_CONV_CHANNELS, 2 * _EMB_DIM)

    def forward(self, x):
        x = x.view([x.size(0), _EMB_REDUCTION_FACTOR, -1])
        layer_outs = []
        for layer in self._layers:
            out = layer.forward(x)
            layer_outs.append(out)
            x = out
        embs = self._output.forward(x)
        return torch.split(embs, _EMB_DIM, dim=1) + (layer_outs,)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self._input = submodules.conv_1d(_EMB_DIM, _CONV_CHANNELS)
        self._layers = nn.ModuleList()
        for i in range(_CONV_LAYERS):
            self._layers.append(submodules.ResNetModule1d(_CONV_CHANNELS, _CONV_CHANNELS, kernel_size=3, pad=1, dilation=2**(i % _CONV_DILATION_CYCLE)))
        self._layers.append(submodules.conv_1d(_CONV_CHANNELS, _EMB_REDUCTION_FACTOR))

    def forward(self, x, encoder_layer_outs):
        x = self._input.forward(x)
        for layer, encoder_layer_out in zip(self._layers, reversed(encoder_layer_outs)):
            x = layer.forward(x + encoder_layer_out)
        x = x.view([x.size(0), -1])
        x = torch.clamp(x, -1.0, 1.0)
        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self._encoder = Encoder()
        self._instr_decoder = Decoder()
        self._vocal_decoder = Decoder()

    def forward(self, instr, vocal):
        # emb1 is pushed to have only instrumental information and emb2 vocals.
        instr_emb1, instr_emb2, _ = self._encoder.forward(instr)
        mixed_emb1, mixed_emb2, encoder_layer_outs = self._encoder.forward(instr + vocal)
        vocal_emb1, vocal_emb2, _ = self._encoder.forward(vocal)

        emb_loss = torch.mean((instr_emb1 - mixed_emb1) ** 2 + (vocal_emb2 - mixed_emb2) ** 2)
        GLOBAL.summary_writer.add_scalar('loss_train/emb_loss', emb_loss, GLOBAL.current_step)
        emb_norm_loss = torch.mean(instr_emb2 ** 2 + vocal_emb1 ** 2)
        GLOBAL.summary_writer.add_scalar('loss_train/emb_norm_loss', emb_norm_loss, GLOBAL.current_step)

        instr_pred = self._instr_decoder.forward(mixed_emb1, encoder_layer_outs)
        instr_loss = torch.mean((instr - instr_pred) ** 2)
        GLOBAL.summary_writer.add_scalar('loss_train/instr_loss', instr_loss, GLOBAL.current_step)
        vocal_pred = self._vocal_decoder.forward(mixed_emb2, encoder_layer_outs)
        vocal_loss = torch.mean((vocal - vocal_pred) ** 2)
        GLOBAL.summary_writer.add_scalar('loss_train/vocal_loss', vocal_loss, GLOBAL.current_step)

        return vocal_pred, emb_loss + emb_norm_loss + instr_loss + vocal_loss

    def predict(self, mixed):
        overflow = mixed.shape[1] % _EMB_REDUCTION_FACTOR
        pad_size = (_EMB_REDUCTION_FACTOR - overflow) % _EMB_REDUCTION_FACTOR
        if pad_size:
          mixed = torch.pad(mixed, [(0, 0), (0, pad_size)])
        _, mixed_emb2, encoder_layer_outs = self._encoder.forward(mixed)
        vocal_pred = self._vocal_decoder.forward(mixed_emb2, encoder_layer_outs)
        return vocal_pred[:, :vocal_pred.shape[1] - pad_size]


class Generator(Network):

    def BuildModel(self):
        return Model()

    def preprocess(self, data):
        data_instrumental, data_vocal = zip(*data)
        instrumental = np.stack(data_instrumental)
        vocal = np.stack(data_vocal)
        assert instrumental.shape == vocal.shape
        assert len(vocal.shape) == 2
        return torch.Tensor(instrumental).cuda(), torch.Tensor(vocal).cuda()

    def forward(self, data):
        return self.model.forward(*data)

    def loss(self, prediction, data):
        return prediction[1]

    def predict(self, data, summary_prefix=''):
        assert len(data[0].shape) == 2 and data[0].shape[0] == 1
        # split into chunks to fit into memory.
        chunk_size = 16000
        context_size = 4000
        assert (chunk_size + 2 * context_size) % _EMB_REDUCTION_FACTOR == 0
        prediction = np.zeros([0])
        for i in range(0, data[0].shape[1], chunk_size):
            start = max(0, i - context_size)
            chunk_end = min(data[0].shape[1], i + chunk_size)
            end = min(data[0].shape[1], chunk_end + context_size)
            data_i = data[0][:, start:end] + data[1][:, start:end]  # mixed = instr + vocal
            prediction_i = self.model.module.predict(data_i)[0].detach().cpu().numpy()
            prediction = np.concatenate((prediction, prediction_i[i - start:chunk_end - start]))
        # since we predicted vocals only, do some math to get off_vocal.
        # prediction = data[0] - prediction
        prediction = np.clip(prediction, -1.0, 1.0)
        return prediction
