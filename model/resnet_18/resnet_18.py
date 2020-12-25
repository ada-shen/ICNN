# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_18.conv_mask import conv_mask

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)#new padding

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = nn.ZeroPad2d(1)#new paddig

    def forward(self, x):
        identity = x
        out = self.pad2d(x) #new padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out) #new padding
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class resnet_18(nn.Module):
    def __init__(self, pretrain_path,num_classes,dropout_rate,losstype,block=BasicBlock,  layers=[2,2,2,2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet_18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.label_num = num_classes
        self.pretrian_path = pretrain_path
        self._norm_layer = norm_layer
        self.losstype = losstype
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)#new padding
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#new padding
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.mask1 = nn.Sequential(
            conv_mask(256* block.expansion, 256* block.expansion, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), labelnum=self.label_num,
                      loss_type=self.losstype, ), )

        self.mask2 = nn.Sequential(
            conv_mask(256 * block.expansion, 256 * block.expansion, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      labelnum=self.label_num,
                      loss_type=self.losstype, ), )

        self.avgpool = nn.Sequential(
            #nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(512* block.expansion, num_classes)

        self.pad2d_1 = nn.ZeroPad2d(1)#new paddig
        self.pad2d_3 = nn.ZeroPad2d(3)#new paddig

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                    nn.init.constant_(m.bn2.weight, 0)

        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def init_weight(self):
        state_dict = torch.load(self.pretrian_path)
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           'fc' not in k and 'layer4.2' not in k}  # 'fc' not in k and 'layer4.1' not in k and
        model_dict = self.state_dict()
        #for k in model_dict:
        #    print(k)
        # print('####################################')
        # print('####################################')
        #for k,v in state_dict.items():
        #    print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict,strict=False)
        torch.nn.init.normal_(self.mask1[0].weight.data, mean=0, std=0.01)
        torch.nn.init.normal_(self.mask2[0].weight.data, mean=0, std=0.01)

        torch.nn.init.normal_(self.fc.weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.fc.bias.data)


    def forward(self, x, label, Iter, density):
        # See note [TorchScript super()]


        x = self.pad2d_3(x) #new padding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2d_1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.mask1[0](x, label, Iter, density)
        x = self.relu1(x)
        x = self.mask2[0](x, label, Iter, density)
        x = self.relu2(x)
        # f_map = x.detach()
        x = self.layer4(x)

        # f_map = x.detach()
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)


        return x
    # def forward(self, x):
    #     return self._forward_impl(x), None

