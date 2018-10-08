import torch
import torch.nn as nn
import numpy as np
import submodules
from network import Network
import utils
from CONSOLE_ARGS import ARGS as FLAGS

class Generator(Network):

    def BuildModel(self):
        n_fft, fft_channels, _ = utils.NFFT()
        # TODO: Use mel.
        # input_channels = FLAGS.n_mels + fft_channels
        input_channels = 2 * fft_channels
        layer_defs = []
        layer_defs.append(nn.Conv1d(input_channels, 256, 1, 1))
        for i in range(50):
            layer_defs.append(submodules.ResNetModule1d(256, 256, 3, 1, 1, 1))
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
            stft_vocal = utils.STFT(data_vocal)
            data_offvocal = d.data[1]
            data_offvocal = utils.Convert16BitToFloat(data_offvocal)
            stft_offvocal = utils.STFT(data_offvocal)
            ret['vocal_mel'].append(utils.MelSpectrogram(stft_vocal))
            ret['vocal_phase'].append(np.angle(stft_vocal))
            ret['offvocal_mel'].append(utils.MelSpectrogram(stft_offvocal))
            ret['offvocal_phase'].append(np.angle(stft_offvocal))
        return ret

    def forward(self, data):
        vocal_stacked = np.concatenate(
            (data['vocal_mel'], data['vocal_phase']), axis=1)
        vocal_stacked = torch.Tensor(vocal_stacked).cuda()
        return self.model.forward(vocal_stacked)

    def loss(self, prediction, data):
        n_fft, fft_channels, _ = utils.NFFT()
        predicted_real = prediction[:, :-fft_channels]
        predicted_imag = prediction[:, -fft_channels:]
        gt_mel = torch.Tensor(data['offvocal_mel']).cuda()
        gt_phase = torch.Tensor(data['offvocal_phase']).cuda()
        gt_real = gt_mel * torch.cos(gt_phase)
        gt_imag = gt_mel * torch.sin(gt_phase)
        loss = torch.mean((predicted_real - gt_real)**2 +
                          (predicted_imag - gt_imag)**2)
        return loss

    def predict(self, data, summary_prefix=''):
        prediction = self.forward(data)
        prediction = prediction.detach().cpu().numpy()
        assert prediction.shape[0] == 1

        n_fft, fft_channels, _ = utils.NFFT()
        predicted_real = prediction[0, :-fft_channels]
        predicted_imag = prediction[0, -fft_channels:]
        predicted_mel = np.sqrt(predicted_real**2 + predicted_imag**2)
        predicted_phase = np.arctan2(predicted_imag, predicted_real)

        self._summary_writer.add_image(summary_prefix + '/gt_mel_onvocal',
                                       utils.PlotMel('gt onvocal', data['vocal_mel'][0]),
                                       self.current_step)
        self._summary_writer.add_image(summary_prefix + '/gt_mel_offvocal',
                                       utils.PlotMel('gt offvocal', data['offvocal_mel'][0]),
                                       self.current_step)
        self._summary_writer.add_image(summary_prefix + '/predicted_mel',
                                       utils.PlotMel('predicted', predicted_mel),
                                       self.current_step)

        predicted_magnitude = utils.InverseMelSpectrogram(predicted_mel)
        predicted_stft = predicted_magnitude * np.exp(1j * predicted_phase)
        result = utils.InverseSTFT(predicted_stft)
        return result

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(Network):

    def BuildModel(self):
        n_fft, fft_channels, _ = utils.NFFT()

        input_channels = 2 * fft_channels
        layer_defs = []
        layer_defs.append(submodules.ResNetModule1d(input_channels, 256, 3, 2, 1, 1))
        layer_defs.append(submodules.ResNetModule1d(256, 256, 3, 1, 1, 1))
        for i in range(4):
            layer_defs.append(submodules.ResNetModule1d(256, 256, 3, 1, 1, 1))
            layer_defs.append(submodules.ResNetModule1d(256, 256, 3, 1, 1, 1))
        layer_defs.append(Flatten())
        layer_defs.append(nn.Linear(15360,2))
        return nn.Sequential(*layer_defs)

    def loss(self, input, labels):

        criterion = nn.CrossEntropyLoss()


        return criterion(input, labels)

    def forward(self, data):
        if type(data) is dict:
            vocal_stacked = np.concatenate(
                (data['offvocal_mel'], data['offvocal_phase']), axis=1)
            data = torch.Tensor(vocal_stacked).cuda()
            #print(data.shape)
        return self.model.forward(data)



