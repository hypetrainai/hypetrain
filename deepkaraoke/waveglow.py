import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import submodules
from network import Network
import utils
from CONSOLE_ARGS import ARGS as FLAGS

_NUM_FLOWS = 12
_N_CHANNELS = 8  # Must be even.
_CONV_LAYERS = 8
_CONV_CHANNELS = 512
_STD_TRAIN = np.sqrt(0.5)
_STD_EVAL = 0.6


class InvertibleConv1x1(nn.Module):
    def __init__(self, c):
        super(InvertibleConv1x1, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

        self.W_inverse = Variable(self.conv.weight.squeeze().inverse())[..., None].cuda()

    def forward(self, x):
        if self.training:
            return self.conv(x)
        else:
            return F.conv1d(x, self.W_inverse, bias=None)


class AffineCoupling(nn.Module):
    def __init__(self):
        super(AffineCoupling, self).__init__()
        layer_defs = []
        layer_defs.append(submodules.conv_1d(_N_CHANNELS // 2, _CONV_CHANNELS, 3, 1, 1, 1, bias=True, wn=True))
        for i in range(_CONV_LAYERS):
            layer_defs.append(submodules.ResNetModule1d(_CONV_CHANNELS, _CONV_CHANNELS, 3, 1, 1, 2**(1+i), bias=True, bn=False, wn=True))
        end = submodules.conv_1d(_CONV_CHANNELS, 2, 3, 1, 1, 1, bias=True)
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end.weight.data.zero_()
        end.bias.data.zero_()
        layer_defs.append(end)
        self.model = nn.Sequential(*layer_defs)

    def forward(self, conditioning, x):
        a, b = torch.split(x, _N_CHANNELS // 2, dim=1)
        log_s, t = torch.split(self.model.forward(a + conditioning), 1, dim=1)
        if self.training:
            b = torch.exp(log_s) * b + t
        else:
            b = (b - t) / torch.exp(log_s)
        return torch.cat((a, b), dim=1), torch.sum(-log_s)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.ModuleList()
        self.coupling = nn.ModuleList()
        for i in range(_NUM_FLOWS):
            self.conv.append(InvertibleConv1x1(_N_CHANNELS))
            self.coupling.append(AffineCoupling())

    def forward(self, conditioning, x):
        assert len(x.size()) == 2
        assert conditioning.size() == x.size()
        x = x.view([x.size()[0], _N_CHANNELS // 2, -1])
        x_a = torch.cat((torch.zeros((x.size()[0], _N_CHANNELS // 2, 1)).cuda(), x[:, :, :-1]), dim=2)
        x = torch.cat((x_a, x), dim=1)
        conditioning = conditioning.view([x.size()[0], _N_CHANNELS // 2, -1])
        total_conv_loss = 0
        total_coupling_loss = 0
        # TODO: early emit.
        for i in range(12):
            x = self.conv[i].forward(x)
            total_conv_loss += -torch.log(torch.abs(torch.det(self.conv[i].conv.weight.squeeze())))
            x, coupling_loss = self.coupling[i].forward(conditioning, x)
            total_coupling_loss += coupling_loss
        _, x = torch.split(x, _N_CHANNELS // 2, dim=1)
        x = x.view([x.size()[0], -1])
        return x, total_conv_loss, total_coupling_loss


class Generator(Network):

    def BuildModel(self):
        return Model()

    def preprocess(self, data):
        on_vocal = np.stack([d.data[0] for d in data])
        off_vocal = np.stack([d.data[1] for d in data])
        assert on_vocal.shape == off_vocal.shape
        assert len(on_vocal.shape) == 2
        if self.training:
            assert _N_CHANNELS % 2 == 0
            assert on_vocal.shape[1] % (_N_CHANNELS // 2)== 0
        else:
            # Pad to multiple of _N_CHANNELS // 2.
            pad_size = (_N_CHANNELS // 2 - on_vocal.shape[1] % (_N_CHANNELS // 2)) % (_N_CHANNELS // 2)
            on_vocal = np.pad(on_vocal, [(0, 0), (0, pad_size)], 'constant')
            off_vocal = np.pad(off_vocal, [(0, 0), (0, pad_size)], 'constant')
        return on_vocal, off_vocal

    def forward(self, data):
        x, total_conv_loss, total_coupling_loss = (
            self.model.forward(torch.Tensor(data[0]).cuda(), torch.Tensor(data[1]).cuda()))
        if self.training:
            self._summary_writer.add_scalar('loss_train/conv', total_conv_loss, self.current_step)
            self._summary_writer.add_scalar('loss_train/coupling', total_coupling_loss, self.current_step)
        loss = total_conv_loss + total_coupling_loss + torch.sum(x * x) / (2 * _STD_TRAIN * _STD_TRAIN)
        loss /= x.size(0) * x.size(1)
        return x, loss

    def loss(self, prediction, data):
        return prediction[1]

    def predict(self, data, summary_prefix=''):
        assert len(data[0].shape) == 2 and data[0].shape[0] == 1
        # replace gt off_vocal data with vector randomly sampled from prior.
        data = (data[0], np.random.normal(0.0, _STD_EVAL, data[0].shape))
        # split into chunks to fit into memory.
        chunk_size = 32000
        assert chunk_size % _N_CHANNELS == 0
        prediction = np.zeros([1, 0])
        for i in range(0, data[0].shape[1], chunk_size):
            end = min(data[0].shape[1], i + chunk_size)
            data_i = [data[0][:, :end], data[1][:, :end]]
            prediction_i = self.forward(data_i)[0].detach().cpu().numpy()
            prediction = np.concatenate((prediction, prediction_i), axis=1)
        return prediction
