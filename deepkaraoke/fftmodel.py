import torch
import torch.nn as nn
from submodules import convbn_1d
from network import Network
import utils


class Simple(Network):
    def __init__(self):
        super(Simple, self).__init__()
        self.sr = 44100
        self.hop_length_ms = 10
        self.window_length_ms = 40
        self.n_mels = 128
        self.fmin = 25

    def BuildModel(self):
        input_channels = self.n_mels + utils.FFTChannels(self.window_length_ms)
        layer_defs = []
        layer_defs.append(convbn_1d(input_channels, 256, 3, 1, 1, 1))
        for i in range(11):
            layer_defs.append(convbn_1d(256, 256, 3, 1, 1, 2**(1 + i)))
        for i in range(11):
            layer_defs.append(convbn_1d(256, 256, 3, 1, 1, 2**(1 + i)))
        layer_defs.append(nn.Conv1d(256, input_channels, 1, 1, 1, 1))
        return nn.Sequential(*layer_defs)

    def preprocess(self, data):
        data_vocal = [d.data[0] for d in data]
        stft_vocal = utils.STFT(data_vocal, self.sr, self.hop_length_ms,
                                self.window_length_ms)
        data_offvocal = [d.data[1] for d in data]
        stft_offvocal = utils.STFT(data_offvocal, self.sr, self.hop_length_ms,
                                   self.window_length_ms)
        return {
            'vocal_mel':
            utils.MelSpectrogram(stft_vocal, self.sr, self.hop_length_ms,
                                 self.window_length_ms, self.n_mels,
                                 self.fmin),
            'vocal_phase':
            np.angle(stft_vocal),
            'offvocal_mel':
            utils.MelSpectrogram(stft_offvocal, self.sr, self.hop_length_ms,
                                 self.window_length_ms, self.n_mels,
                                 self.fmin),
            'offvocal_phase':
            np.angle(stft_offvocal),
        }

    def forward(self, data):
        vocal_stacked = np.concatenate(
            data['vocal_mel'], data['vocal_phase'], axis=1)
        vocal_stacked = torch.Tensor(vocal_stacked).cuda()
        return self.model.forward(vocal_stacked)

    def loss(self, prediction, data):
        predict_mel = prediction[:, :self.n_mels]
        loss_mel = torch.mean((predict_mel - data['offvocal_mel'])**2)

        predict_phase = prediction[:, self.n_mels:]
        loss_phase = torch.mean(1 -
                                np.cos(predict_phase - data['offvocal_phase']))

        return loss_mel + loss_phase

    def predict(self, data):
        prediction = self.forward(data)
        predict_mel = prediction[:, :self.n_mels]
        predict_phase = prediction[:, self.n_mels:]
        predict_magnitude = utils.InverseMelSpectrogram(
            predict_mel, self.sr, self.window_length_ms, self.n_mels,
            self.fmin)
        predict_stft = predict_magnitude * exp(1j * predict_phase)
        return utils.InverseSTFT(predict_stft, self.sr, self.hop_length_ms,
                                 self.window_length_ms)
