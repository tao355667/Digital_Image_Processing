# -*- coding: utf-8 -*-
# %% 形态学处理
import cv2 as cv  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# matplotlib中文字体
font = {'family': 'SimHei', 'weight': 'bold', 'size': '16'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
# %% 读图，转灰度
img = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/vegetation_22.png', cv.IMREAD_COLOR)  # BGR原始图
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB原始图
imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)   # RGB转灰度
# %%预处理,分割
imgBlur = cv.blur(imgGray, (3, 3))  # 均值滤波
th1ret, th1 = cv.threshold(
    imgBlur, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
plt.figure('阈值分割，形态学')
plt.subplot(111), plt.imshow(th1, 'gray'), plt.title(
    "3x3均值滤波后，OTSU(T={})".format(th1ret)), plt.axis('off')
plt.show()

