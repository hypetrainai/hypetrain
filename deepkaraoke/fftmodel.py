import torch
import torch.nn as nn
import numpy as np
from submodules import convbn_1d
from network import Network
import utils


class Simple(Network):

    def __init__(self, summary_writer):
        self.sr = 44100
        self.hop_length_ms = 10
        self.window_length_ms = 40
        self.n_mels = 128
        self.fmin = 0
        super(Simple, self).__init__(summary_writer)

    def BuildModel(self):
        n_fft, _ = utils.NFFT(self.sr, self.window_length_ms)
        fft_channels = (n_fft + 1) // 2 + 1
        input_channels = self.n_mels + fft_channels
        layer_defs = []
        layer_defs.append(convbn_1d(input_channels, 256, 3, 1, 1, 1))
        for i in range(11):
            layer_defs.append(convbn_1d(256, 256, 3, 1, 1, 1))
        layer_defs.append(nn.Conv1d(256, input_channels, 1, 1))
        return nn.Sequential(*layer_defs)

    def preprocess(self, data):
        ret = {
            'vocal_mel': [],
            'vocal_phase': [],
            'offvocal_mel': [],
            'offvocal_phase': [],
        }
        for d in data:
            data_vocal = d.data[0]
            data_vocal = utils.Convert16BitToFloat(data_vocal)
            stft_vocal = utils.STFT(
                data_vocal, self.sr, self.hop_length_ms, self.window_length_ms)
            data_offvocal = d.data[1]
            data_offvocal = utils.Convert16BitToFloat(data_offvocal)
            stft_offvocal = utils.STFT(
                data_offvocal, self.sr, self.hop_length_ms, self.window_length_ms)
            ret['vocal_mel'].append(
                utils.MelSpectrogram(
                    stft_vocal, self.sr, self.hop_length_ms,
                    self.window_length_ms, self.n_mels, self.fmin))
            ret['vocal_phase'].append(np.angle(stft_vocal))
            ret['offvocal_mel'].append(
                utils.MelSpectrogram(
                    stft_offvocal, self.sr, self.hop_length_ms,
                    self.window_length_ms, self.n_mels, self.fmin))
            ret['offvocal_phase'].append(np.angle(stft_offvocal))
        return ret

    def forward(self, data):
        vocal_stacked = np.concatenate(
            (data['vocal_mel'], data['vocal_phase']), axis=1)
        vocal_stacked = torch.Tensor(vocal_stacked).cuda()
        return self.model.forward(vocal_stacked)

    def loss(self, prediction, data):
        predicted_mel = prediction[:, :self.n_mels]
        gt_mel = torch.Tensor(data['offvocal_mel']).cuda()
        loss_mel = torch.mean((predicted_mel - gt_mel)**4)

        predicted_phase = prediction[:, self.n_mels:]
        gt_phase = torch.Tensor(data['offvocal_phase']).cuda()
        loss_phase = torch.mean(1 - torch.cos(predicted_phase - gt_phase))

        summary_prefix = 'loss_train' if self.training else 'loss_test'
        self._summary_writer.add_scalar(summary_prefix + '/mel', loss_mel, self.current_step)
        self._summary_writer.add_scalar(summary_prefix + '/phase', loss_phase, self.current_step)

        # TODO: predict phase.
        return loss_mel + 0 * loss_phase

    def predict(self, data):
        prediction = self.forward(data)
        prediction = prediction.detach().cpu().numpy()
        assert prediction.shape[0] == 1
        predicted_mel = prediction[0, :self.n_mels]
        predicted_phase = prediction[0, self.n_mels:]
        # TODO: predict phase.
        predicted_phase = data['vocal_phase'][0]

        summary_prefix = 'train' if self.training else 'test'
        self._summary_writer.add_image(summary_prefix + '/gt_onvocal',
                                       utils.PlotMel('gt onvocal', data['vocal_mel'][0]),
                                       self.current_step)
        self._summary_writer.add_image(summary_prefix + '/gt_offvocal',
                                       utils.PlotMel('gt offvocal', data['offvocal_mel'][0]),
                                       self.current_step)
        self._summary_writer.add_image(summary_prefix + '/predicted',
                                       utils.PlotMel('predicted', predicted_mel),
                                       self.current_step)

        predicted_magnitude = utils.InverseMelSpectrogram(
            predicted_mel, self.sr, self.window_length_ms, self.n_mels, self.fmin)
        predicted_stft = predicted_magnitude * np.exp(1j * predicted_phase)
        result = utils.InverseSTFT(
            predicted_stft, self.sr, self.hop_length_ms, self.window_length_ms)
        return result
