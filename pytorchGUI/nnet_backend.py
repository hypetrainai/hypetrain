import matplotlib.pyplot as plt
import numpy as np
import imageio
from IPython import display
from sklearn.model_selection import KFold

from os import listdir,mkdir,rmdir
from os.path import join,isdir

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader













class test_net(nn.Module):
    def __init__(self, c=3, img_size=32):
        super(test_net, self).__init__()
        
        self.mp = nn.ReLU()
        self.c1 = nn.Conv2d(3, out_channels=96,kernel_size=3,stride=1,padding=1)
        self.b1 = nn.BatchNorm2d(96)
        self.p1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.c2 = nn.Conv2d(96, out_channels=256,kernel_size=3,stride=1,padding=1)
        self.b2 = nn.BatchNorm2d(256)
        self.p2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.c3 = nn.Conv2d(256, out_channels=512,kernel_size=3,stride=1,padding=1)
        self.b3 = nn.BatchNorm2d(512)
        self.c4 = nn.Conv2d(512, out_channels=512,kernel_size=3,stride=1,padding=1)
        self.b4 = nn.BatchNorm2d(512)
        self.c5 = nn.Conv2d(512, out_channels=384,kernel_size=3,stride=1,padding=1)
        self.b5 = nn.BatchNorm2d(384)
        self.p5 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.flatten_size = 384#int(384 * (float(img_size) / 2 / 2 / 2) ** 2)
        self.a5 = nn.AvgPool2d(int(32 / 2 / 2 / 2))
        
        self.f6 = nn.Linear(self.flatten_size, out_features=1024)
        self.b6 = nn.BatchNorm1d(1024)
        self.f8 = nn.Linear(1024, 10)
        
    def forward(self, x):
        l1 = self.p1(self.mp(self.b1(self.c1(x))))
        l2 = self.p2(self.mp(self.b2(self.c2(l1))))
        l3 = self.mp(self.b3(self.c3(l2)))
        l4 = self.mp(self.b4(self.c4(l3)))
        l5 = self.p5(self.mp(self.b5(self.c5(l4))))
        f = self.a5(l5).view(-1, self.flatten_size)
        #f  = l5.view(-1, self.flatten_size)
        l6 = self.mp(self.b6(self.f6(f)))
        l8 = self.mp(self.f8(l6))
        
        return l1,l2,l3,l4,l5,l6,l8












class test_dataset(Dataset):
    def __init__(self, filepath='/home/darvin/Data/cifar-10-train'):
        self.datapath = filepath
        
    def __len__(self):
        length = 0
        for folder in listdir(self.datapath):
            path_imgs = join(self.datapath, folder)
            length += len(listdir(path_imgs))
        return length
    
    def __getitem__(self, idx):
        lengths = []
        for folder in listdir(self.datapath):
            path_imgs = join(self.datapath, folder)
            lengths.append(len(listdir(path_imgs)))
        for ii in range(len(lengths)):
            if idx < lengths[ii]:
                break
            idx -= lengths[ii]
        path_imgs = join(self.datapath, listdir(self.datapath)[ii])
        path_img = join(path_imgs, listdir(path_imgs)[idx])
        img = np.array(imageio.imread(path_img))
        img = np.transpose(img, [2,0,1])
        lab = ii
        return {'img': img, 'lab': lab}









def run_fx():
    dataset = test_dataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=6)

    lr = 0.001
    weight_decay=0.0001

    net = test_net().cuda()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()

    for i_batch,sample_batch in enumerate(dataloader):
        opt.zero_grad()
        data = sample_batch['img'].cuda().float()
        labs = sample_batch['lab'].cuda().long()
        pred = net(data)
        L = loss(pred[-1], labs)
        L.backward()
        opt.step()
        print(i_batch, L)






