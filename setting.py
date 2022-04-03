#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: setting.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:53:36
@LastEditTime: 2022-04-03 18:39:17
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import torch
import os

# PATH
DATASET_PATH = "Dataset_BUSI"
MODEL_PATH = "models"

# TORCH
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL = 2
VAL_INTERVAL = 1

# HYPER-PARAMETER
INPUT_SIZE = 128

MAX_EPOCH = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5

# PREDICT
PREDICT_MODEL_PATH = "MedicalImageDiagnosis\models\AlexNet_fold_0.pt"