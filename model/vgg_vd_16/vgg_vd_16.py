# -*- coding: utf-8 -*-

import h5py
import math
import copy
import scipy.io as io
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg_vd_16.conv_mask import conv_mask



class vgg_vd_16(nn.Module):
    def __init__(self, pretrain_path, label_num, dropoutrate, losstype):
        super(vgg_vd_16, self).__init__()
        self.pretrian_path = pretrain_path
        self.dropoutrate = dropoutrate
        self.label_num = label_num
        self.losstype = losstype
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        self.maxpool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        self.maxpool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        self.mask1 = nn.Sequential(
            conv_mask(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), labelnum=self.label_num, loss_type = self.losstype, ), )
        self.maxpool5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.mask2 = nn.Sequential(
            conv_mask(512, 4096, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), labelnum=self.label_num, loss_type = self.losstype, ), )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True), )
        self.line = nn.Sequential(
            nn.Dropout2d(p=self.dropoutrate),
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropoutrate),
            nn.Conv2d(4096, self.label_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)), )
        self.init_weight()

    def init_weight(self):
        data = loadmat(self.pretrian_path)
        w, b = data['layers'][0][0][0]['weights'][0][0]
        self.conv1[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv1[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][2][0]['weights'][0][0]
        self.conv1[2].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv1[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][5][0]['weights'][0][0]
        self.conv2[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv2[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][7][0]['weights'][0][0]
        self.conv2[2].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv2[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][10][0]['weights'][0][0]
        self.conv3[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv3[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][12][0]['weights'][0][0]
        self.conv3[2].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv3[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][14][0]['weights'][0][0]
        self.conv3[4].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv3[4].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][17][0]['weights'][0][0]
        self.conv4[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv4[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][19][0]['weights'][0][0]
        self.conv4[2].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv4[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][21][0]['weights'][0][0]
        self.conv4[4].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv4[4].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][24][0]['weights'][0][0]
        self.conv5[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv5[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][26][0]['weights'][0][0]
        self.conv5[2].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv5[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][28][0]['weights'][0][0]
        self.conv5[4].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv5[4].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        torch.nn.init.normal_(self.mask1[0].weight.data, mean=0, std=0.01)
        torch.nn.init.normal_(self.mask2[0].weight.data, mean=0, std=0.01)

        torch.nn.init.normal_(self.line[1].weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.line[1].bias.data)
        torch.nn.init.normal_(self.line[4].weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.line[4].bias.data)

    def forward(self, x, label, Iter, density):
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool4(x)

        x = self.conv5(x)

        x = self.mask1[0](x, label, Iter, density)
        x = self.maxpool5(x)

        x = self.mask2[0](x, label, Iter, density)
        x = self.relu(x)

        #x = x.view(x.size(0),-1) ##
        #x = x.unsqueeze(2)
        #x = x.unsqueeze(3)

        x = self.line(x)
        return x





