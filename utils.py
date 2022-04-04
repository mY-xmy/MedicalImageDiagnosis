#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: utils.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:53:12
@LastEditTime: 2022-04-04 16:59:22
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import os
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import setting

def parse_label(img_path):
    filename = img_path.split("/")[-1]
    label = filename.split("_")[0]
    return label

def label_encode(x):
    label_map = {"benign": 1, "malignant": 2, "normal" : 0}
    return label_map[x]

def label_decode(x):
    label_map = {1: "benign", 2: "malignant", 0: "normal"}
    return label_map[x]

def get_train_val_data(data_dir, k=5, isFold=True):
    image_files = [image_file for image_file in os.listdir(data_dir)]
    labels = [label_encode(image_file.split("_")[0]) for image_file in image_files]
    image_paths = [os.path.join(data_dir, image_file) for image_file in image_files]

    if isFold:
        skf = StratifiedKFold(n_splits=k, random_state=6001, shuffle=True)
        for train_index, test_index in skf.split(image_files, labels):
            train_image_files = image_paths[train_index]
            val_image_files = image_paths[test_index]
            train_labels = labels[image_files]
            val_labels = labels[test_index]
            yield train_image_files, val_image_files, train_labels, val_labels
    else:
        train_image_files, val_image_files, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=0, stratify=labels)
        yield train_image_files, val_image_files, train_labels, val_labels

class BUSI_Dataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.image_paths = paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index): 
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels[index])
        return image, label

    def __len__(self): 
        return len(self.image_paths)

train_transform = transforms.Compose([
    transforms.Resize((setting.INPUT_SIZE, setting.INPUT_SIZE)),
    transforms.RandomResizedCrop(size=setting.INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
])

valid_transform = transforms.Compose([
    transforms.Resize((setting.INPUT_SIZE, setting.INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
])

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss