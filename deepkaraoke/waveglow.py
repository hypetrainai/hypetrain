import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import submodules
from network import Network
import utils
from GLOBALS import FLAGS, GLOBAL

_NUM_FLOWS = 12
_N_CHANNELS = 8
assert _N_CHANNELS % 8 == 0
_COUPLING_LAYERS = 8
_COUPLING_CHANNELS = 256
_STD_TRAIN = 1.0
_STD_EVAL = 0.6


class InvertibleConv1x1(nn.Module):
    def __init__(self, c):
        super(InvertibleConv1x1, self).__init__()
        self.channels = c
        # Sample a random orthonormal matrix to initialize weights
        w_init = np.linalg.qr(np.random.randn(c, c))[0].astype(np.float32)
        # Ensure determinant is 1.0 not -1.0
        if np.linalg.det(w_init) < 0:
          w_init[:, 0] = -1 * w_init[:, 0]
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def forward(self, x, reverse=False):
        weight = self.weight
        if reverse:
            weight = torch.inverse(weight)
        weight = weight.view(self.channels, self.channels, 1)
        return F.conv1d(x, weight)


class AffineCoupling(nn.Module):
    def __init__(self, channels):
        super(AffineCoupling, self).__init__()
        self.channels = channels
        assert self.channels % 2 == 0
        self.start = submodules.conv_1d(self.channels // 2, _COUPLING_CHANNELS)
        self.cond_proj = submodules.conv_1d(self.channels, _COUPLING_CHANNELS)
        self.in_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        for i in range(_COUPLING_LAYERS):
            self.in_layers.append(submodules.conv_1d(_COUPLING_CHANNELS, 2 * _COUPLING_CHANNELS, kernel_size=3, pad=1, dilation=2**i))
            self.cond_layers.append(submodules.conv_1d(_COUPLING_CHANNELS, 2 * _COUPLING_CHANNELS, kernel_size=3, pad=1, dilation=2**i))
            self.res_layers.append(submodules.conv_1d(_COUPLING_CHANNELS, 2 * _COUPLING_CHANNELS))
        self.end = submodules.conv_1d(_COUPLING_CHANNELS, self.channels, bias=True)
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability.
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, conditioning, x, reverse=False):
        a, b = torch.split(x, self.channels // 2, dim=1)
        conditioning = self.cond_proj(conditioning)
        input = self.start(a)
        output = torch.zeros_like(input)
        for i in range(_COUPLING_LAYERS):
            in_act = self.in_layers[i](input) + self.cond_layers[i](conditioning)
            res = torch.tanh(in_act[:, :_COUPLING_CHANNELS, :]) * torch.sigmoid(in_act[:, _COUPLING_CHANNELS:, :])
            res = self.res_layers[i](res)
            input = input + res[:, :_COUPLING_CHANNELS, :]
            output = output + res[:, _COUPLING_CHANNELS:, :]
        output = self.end(output)

        log_s, t = torch.split(output, self.channels // 2, dim=1)
        log_s = torch.clamp(log_s, -2, 2)
        if not reverse:
            b = torch.exp(log_s) * b + t
        else:
            b = (b - t) / torch.exp(log_s)
        return torch.cat((a, b), dim=1), torch.sum(-log_s)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.ModuleList()
        self.coupling = nn.ModuleList()
        self.emit_layers = []
        self.emit_channels = _N_CHANNELS // 4
        assert self.emit_channels % 2 == 0
        remaining_channels = _N_CHANNELS
        for i in range(_NUM_FLOWS):
            if i in self.emit_layers:
                remaining_channels -= self.emit_channels
            self.conv.append(InvertibleConv1x1(remaining_channels))
            self.coupling.append(AffineCoupling(remaining_channels))

    def forward(self, conditioning, x, reverse=False):
        assert len(x.size()) == 2
        assert conditioning.size() == x.size()
        x = x.view([x.size(0), _N_CHANNELS, -1])
        conditioning = conditioning.view([x.size(0), _N_CHANNELS, -1])
        total_conv_loss = 0
        total_coupling_loss = 0
        outputs = []
        if not reverse:
            for i in range(_NUM_FLOWS):
                if i in self.emit_layers:
                    outputs.append(x[:, :self.emit_channels, :])
                    x = x[:, self.emit_channels:, :]
                out = self.conv[i].forward(x)
                total_conv_loss += -x.size(1) * x.size(2) * torch.logdet(self.conv[i].weight)
                if FLAGS.debug:
                  with torch.no_grad():
                      pred = self.conv[i].forward(out, reverse=True)
                      diff = np.amax(np.abs((x - pred).detach().cpu().numpy()))
                      GLOBAL.summary_writer.add_scalar('reverse/conv_%d' % i, diff, GLOBAL.current_step)
                x = out
                out, coupling_loss = self.coupling[i].forward(conditioning, x)
                total_coupling_loss += coupling_loss
                if FLAGS.debug:
                  with torch.no_grad():
                      pred, _ = self.coupling[i].forward(conditioning, out, reverse=True)
                      diff = np.amax(np.abs((x - pred).detach().cpu().numpy()))
                      GLOBAL.summary_writer.add_scalar('reverse/coupling_%d' % i, diff, GLOBAL.current_step)
                x = out
        else:
            remaining_channels = _N_CHANNELS - len(self.emit_layers) * self.emit_channels
            assert remaining_channels > 0
            x_remaining = x[:, :-remaining_channels, :]
            x = x[:, -remaining_channels:, :]
            for i in reversed(range(_NUM_FLOWS)):
                x, _ = self.coupling[i].forward(conditioning, x, reverse=True)
                x = self.conv[i].forward(x, reverse=True)
                if i in self.emit_layers:
                    x = torch.cat([x_remaining[:, -self.emit_channels:, :], x], dim=1)
                    x_remaining = x_remaining[:, :-self.emit_channels, :]
        outputs.append(x)

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view([outputs.size(0), -1])
        return outputs, total_conv_loss, total_coupling_loss


class Generator(Network):

    def BuildModel(self):
        model = Model()

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier(m.weight.data)

        model.apply(init_weights)
        return model

    def preprocess(self, data):
        data_instrumental, data_vocal = zip(*data)
        instrumental = np.stack(data_instrumental)
        vocal = np.stack(data_vocal)
        assert instrumental.shape == vocal.shape
        assert len(vocal.shape) == 2
        if self.training:
            assert vocal.shape[1] % _N_CHANNELS == 0
        else:
            # Pad to multiple of _N_CHANNELS.
            pad_size = (_N_CHANNELS - vocal.shape[1] % _N_CHANNELS) % _N_CHANNELS
            instrumental = np.pad(instrumental, [(0, 0), (0, pad_size)], 'constant')
            vocal = np.pad(vocal, [(0, 0), (0, pad_size)], 'constant')
        return instrumental + vocal, vocal

    def forward(self, data, reverse=False):
        x, total_conv_loss, total_coupling_loss = (
            self.model.forward(torch.Tensor(data[0]).cuda(), torch.Tensor(data[1]).cuda(), reverse))
        if self.training and not reverse:
            GLOBAL.summary_writer.add_scalar('loss_train/conv', total_conv_loss, GLOBAL.current_step)
            GLOBAL.summary_writer.add_scalar('loss_train/coupling', total_coupling_loss, GLOBAL.current_step)
            if FLAGS.debug:
              with torch.no_grad():
                pred, _, _ = self.model.forward(torch.Tensor(data[0]).cuda(), x, reverse=True)
                diff = np.amax(np.abs(data[1] - pred.detach().cpu().numpy()))
                GLOBAL.summary_writer.add_scalar('reverse/all', diff, GLOBAL.current_step)
        loss = total_conv_loss + total_coupling_loss
        loss += torch.sum(x * x) / (2 * _STD_TRAIN * _STD_TRAIN)
        loss /= x.size(0) * x.size(1)
        return x, loss

    def loss(self, prediction, data):
        return prediction[1]

    def predict(self, data, summary_prefix=''):
        assert len(data[0].shape) == 2 and data[0].shape[0] == 1
        # replace gt off_vocal data with vector randomly sampled from prior.
        data = (data[0], np.random.normal(0.0, _STD_EVAL, data[0].shape))
        # split into chunks to fit into memory.
        chunk_size = 16000
        context_size = 4000
        assert (chunk_size + 2 * context_size) % _N_CHANNELS == 0
        prediction = np.zeros([0])
        for i in range(0, data[0].shape[1], chunk_size):
            start = max(0, i - context_size)
            chunk_end = min(data[0].shape[1], i + chunk_size)
            end = min(data[0].shape[1], chunk_end + context_size)
            data_i = [data[0][:, start:end], data[1][:, start:end]]
            prediction_i = self.forward(data_i, reverse=True)[0][0].detach().cpu().numpy()
            prediction = np.concatenate((prediction, prediction_i[i - start:chunk_end - start]))
        # since we predicted vocals only, do some math to get off_vocal.
        prediction = data[0] - prediction
        prediction = np.clip(prediction, -1.0, 1.0)
        return prediction
