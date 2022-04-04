#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: main.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:51:56
@LastEditTime: 2022-04-04 14:37:19
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import sys
from PyQt5.QtWidgets import QApplication
import torch
import torch.nn as nn
import torch.nn.functional as F
from gui import Diagnosis

from utils import valid_transform
from PIL import Image
from models.CNN import AlexNet
import setting

import pdb

def load_model():
    model = torch.load(setting.PREDICT_MODEL_PATH).cpu()
    model.eval()
    return model

def predict_func(model):
    def predict(image):
        # load image
        image = Image.open(image).convert("RGB")
        image = valid_transform(image)
        image = image.unsqueeze(0)
        # predict image
        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output.squeeze())
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()
        prob = prob.numpy()
        return (predicted, prob) # return (预测类别, 概率)
    return predict

def main():
    model = load_model()
    app = QApplication(sys.argv)
    window = Diagnosis(predict_func=predict_func(model))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    sys.exit(main())