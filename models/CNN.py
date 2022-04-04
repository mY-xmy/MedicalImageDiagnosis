#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: CNN.py
@Author: Xu Mingyu
@Date: 2022-03-26 21:33:28
@LastEditTime: 2022-04-04 17:11:26
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import ChannelAttention, SpatialAttention

class CNN_block(nn.Module):
    def __init__(self, in_channel, out_channel, with_attention=False) -> None:
        super().__init__()
        self.with_attention = with_attention
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.with_attention:
            self.ca = ChannelAttention(out_channel)
            self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.with_attention:
            x = self.ca(x) * x
            x = self.sa(x) * x
        x = self.maxpool(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_class, num_layer, channels, with_attention=False):
        super(CNN, self).__init__()
        assert num_layer == len(channels)
        self.with_attention = with_attention
        block_list = []
        for i in range(num_layer):
            if i == 0:
                block_list.append(CNN_block(in_channel=3, out_channel=channels[i], with_attention=with_attention))
            else:
                block_list.append(CNN_block(in_channel=channels[i-1], out_channel=channels[i]))
        self.features = nn.Sequential(*block_list)
        if self.with_attention:
            self.ca = ChannelAttention(channels[-1])
            self.sa = SpatialAttention()
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(64, num_class),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        bs = x.shape[0]
        x = self.features(x)
        if self.with_attention:
            x = self.ca(x) * x
            x = self.sa(x) * x
        x = self.pooling(x)
        x = x.view(bs, -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, class_num, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # [64, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [64, 27, 27]
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # [192, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [192, 13, 13]
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # [384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [256, 6, 6]
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(64, class_num),
        )
        if init_weights:
            self._initialize_weights()
        ### End your code ###
        
        
    def forward(self, x):
        bs = x.shape[0]
        x = self.features(x)
        x = x.view(bs, -1)
        # x = torch.flatten(self.pooling(x), start_dim=1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)