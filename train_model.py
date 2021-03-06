#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: train_model.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:53:31
@LastEditTime: 2022-04-04 17:59:27
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils import get_train_val_data, BUSI_Dataset, train_transform, valid_transform
from utils import FocalLoss
import setting

from models.CNN import AlexNet, CNN
from models.WideResNet import Wide_ResNet
from models.resnet_cbam import resnext50_32x4d

import pdb

def main():
    # load dataset
    for fold, (train_image_files, val_image_files, train_labels, val_labels) in enumerate(get_train_val_data(setting.DATASET_PATH, isFold=False)):
        print("Fold: %d" % fold)
        train_data = BUSI_Dataset(paths=train_image_files, labels=train_labels, transform=train_transform)
        valid_data = BUSI_Dataset(paths=val_image_files, labels=val_labels, transform=valid_transform)

        train_loader = DataLoader(dataset=train_data, batch_size=setting.BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=1)

        # AlexNet
        # model = AlexNet(3, init_weights=True)

        # CNN with CBAM Attention
        model = CNN(num_class=3, num_layer=setting.NUM_LAYER, channels=setting.CHANNELS, with_attention=setting.WITH_ATTENTION)
        
        # Wide ResNet
        # model = Wide_ResNet(16, 10, 0.2, 3)

        # ResNet with CBAM Attention
        # model = resnext50_32x4d(pretrained=True)

        model.to(setting.DEVICE)

        criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(class_num=3)
        optimizer = optim.AdamW(model.parameters(), lr=setting.LR, weight_decay=setting.WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
        
        max_acc = 0.
        reached = 0  # which epoch reached the max accuracy
        best_confusion_mat = None

        for epoch in range(1, setting.MAX_EPOCH + 1):
            loss_mean = 0.
            model.train()
            for i, data in enumerate(train_loader):
                inputs, labels = data
                # labels = labels.flatten()
                inputs = inputs.to(setting.DEVICE)
                labels = labels.to(setting.DEVICE)
                outputs = model(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_mean += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_acc = (predicted == labels).sum() / labels.shape[0]

                if (i+1) % setting.LOG_INTERVAL == 0:
                    loss_mean = loss_mean / setting.LOG_INTERVAL
                    print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, setting.MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc))
                    loss_mean = 0.
            
            scheduler.step()
            if epoch % setting.VAL_INTERVAL == 0:
                model.eval()
                correct = 0
                total = 0
                loss_val = 0.
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        inputs, labels = data
                        inputs = inputs.to(setting.DEVICE)
                        labels = labels.to(setting.DEVICE)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        loss_val += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.shape[0]
                        correct += (predicted == labels).sum()
                        y_true.append(labels)
                        y_pred.append(predicted)
                    
                    loss_val = loss_val / total
                    val_acc = correct / total

                    y_true = torch.cat(y_true).cpu().numpy()
                    y_pred = torch.cat(y_pred).cpu().numpy()
                    matrix = confusion_matrix(y_true, y_pred)

                    if val_acc > max_acc:
                        max_acc = val_acc
                        reached = epoch
                        best_confusion_mat = matrix
                        torch.save(model, os.path.join(setting.MODEL_PATH, "5CNN_fold_%d.pt" % fold))

                    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                        epoch, setting.MAX_EPOCH, i+1, len(valid_loader), loss_val, val_acc))
                    print("Confusion Matrix:\n", matrix)
                    print("\n")

        print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))
        print("Best Confustion Matrix: \n", best_confusion_mat)

if __name__ == '__main__':
    main()