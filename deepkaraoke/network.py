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

    def predict(self, data):
        return forward(data)

    def loss(self, prediction, data):
        raise NotImplementedError()



