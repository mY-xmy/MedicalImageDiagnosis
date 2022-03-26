#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: gui.py
@Author: Xu Mingyu
@Date: 2022-03-26 19:53:07
@LastEditTime: 2022-03-26 23:09:37
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton,QLabel,QTextEdit,QFileDialog,QHBoxLayout,QVBoxLayout,QSplitter,QComboBox,QSpinBox
from PyQt5.Qt import QWidget
from PyQt5 import QtCore, QtGui

from utils import parse_label, label_decode

class Diagnosis(QMainWindow):
    def __init__(self, predict_func):
        super(Diagnosis, self).__init__()
        self.init_ui()
        self.predict_func = predict_func
        self.img_path = None

    def init_ui(self):
        # self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640,520)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('医疗图像诊断')

        self.lb_name_1 = QLabel('实际类别：', self)
        self.lb_name_1.setGeometry(500, 20, 120, 35)

        self.lb_label = QLabel(self)
        self.lb_label.setGeometry(500, 60, 100, 35)

        self.lb_name_2 = QLabel('识别结果：', self)
        self.lb_name_2.setGeometry(500, 100, 120, 35)

        self.lb_result = QLabel(self)
        self.lb_result.setGeometry(500, 140, 100, 35)

        self.lb_name_3 = QLabel('识别概率：', self)
        self.lb_name_3.setGeometry(500, 180, 100, 35)

        self.lb_confidence = QLabel(self)
        self.lb_confidence.setGeometry(500, 220, 100, 50)

        self.lb_picture = QLabel('待载入图片', self)
        self.lb_picture.setGeometry(10, 20, 480, 480)
        self.lb_picture.setStyleSheet("QLabel{background:gray;}"
                                 "QLabel{color:rgb(0,0,0,120);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        self.lb_picture.setAlignment(QtCore.Qt.AlignCenter)
        
        # self.edit = QTextEdit(self)
        # self.edit.setGeometry(500, 220, 100, 60)

        self.btn_select = QPushButton('选择图片',self)
        self.btn_select.setGeometry(500, 320, 100, 30)
        self.btn_select.clicked.connect(self.select_image)

        self.btn_recog = QPushButton('识别图片',self)
        self.btn_recog.setGeometry(500, 370, 100, 30)
        self.btn_recog.clicked.connect(self.on_btn_Recognize_Clicked)

        # self.btn = QPushButton('返回',self)
        # self.btn.setGeometry(500, 420, 100, 30)
        # self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出',self)
        self.btn_exit.setGeometry(500, 470, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_image(self):
        # global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.lb_picture.width(), self.lb_picture.height())
        self.lb_picture.setPixmap(jpg)
        self.img_path = imgName
        # parse ground-truth label
        label = parse_label(self.img_path)
        self.lb_label.setText(label)
        self.lb_result.clear()
        self.lb_confidence.clear()

    def on_btn_Recognize_Clicked(self):
        # global fname
        # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        # savePath = fname
        # # 加载图像
        # img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        # img = img.convert('L')
        # x = keras.preprocessing.image.img_to_array(img)
        # x = abs(255-x)
        # #x = x.reshape(28,28)
        # x = np.expand_dims(x, axis=0)
        # x=x/255.0
        # new_model = keras.models.load_model('./my_model.h5')
        # prediction = new_model.predict(x)
        # output = np.argmax(prediction, axis=1)
        # self.edit.setText('识别的手写数字为:' + str(output[0]))
        predicted, prob = self.predict_func(self.img_path)
        # prob = [0.1, 0.7, 0.2]
        print(predicted, prob)
        self.lb_result.setText(label_decode(predicted))
        self.lb_confidence.setText("normal: {:.2f} \nbenign: {:.2f} \nmalignant: {:.2f}".format(prob[0], prob[1], prob[2]))
        
    def Quit(self):
        self.close()

    # def slot_btn_function(self):
    #     self.hide()
    #     self.f = FirstUi()
    #     self.f.show()