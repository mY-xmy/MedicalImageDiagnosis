#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: utils.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:53:12
@LastEditTime: 2022-03-26 23:09:28
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import os
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
        train_image_files, val_image_files, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=6001, stratify=labels)
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
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
])