import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.DataParallel(self.BuildModel()).cuda()

    def BuildModel(self):
        raise NotImplementedError()

    def preprocess(self, data):
        return data

    def forward(self, data):
        raise NotImplementedError()

    def loss(self, prediction, data):
        raise NotImplementedError()

    def predict(self, data, summary_prefix=''):
        prediction = prediction.detach().cpu().numpy()
        assert prediction.shape[0] == 1
        return prediction
