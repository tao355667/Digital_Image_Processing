# -*- coding: utf-8 -*-
# 读入河流图像，预处理
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
    './CourseDesignData/dataset-river/train/4_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB原始图
imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)   # RGB转灰度
# 原始图+灰度图
plt.figure('原始图+灰度图')
plt.subplot(121), plt.imshow(img), plt.title('原始图'), plt.axis('off')
plt.subplot(122), plt.imshow(
    imgGray, 'gray'), plt.title('灰度图'), plt.axis('off')
plt.show()

# %% 灰度图预处理
imgEqu = cv.equalizeHist(imgGray)  # 均衡化
# 原始图+灰度图
plt.figure('直方图均衡化')
plt.subplot(221), plt.imshow(
    imgGray, 'gray'), plt.title('灰度图'), plt.axis('off')
plt.subplot(222), plt.hist(imgGray.ravel(), 256, [
    0, 256]), plt.title('灰度直方图')
plt.subplot(223), plt.imshow(imgEqu, 'gray'), plt.title('均衡化'), plt.axis('off')
plt.subplot(224), plt.hist(imgEqu.ravel(), 256,
                           [0, 256]), plt.title('均衡化后灰度直方图')
plt.show()
imgBlur3 = cv.blur(imgGray, (3, 3))  # 均值滤波
imgBlur5 = cv.blur(imgGray, (5, 5))  # 均值滤波
imgBlur7 = cv.blur(imgGray, (7, 7))  # 均值滤波
plt.figure('均值滤波')
plt.subplot(221), plt.imshow(
    imgGray, 'gray'), plt.title('灰度图'), plt.axis('off')
plt.subplot(222), plt.imshow(
    imgBlur3, 'gray'), plt.title('3x3均值滤波'), plt.axis('off')
plt.subplot(223), plt.imshow(
    imgBlur5, 'gray'), plt.title('5x5均值滤波'), plt.axis('off')
plt.subplot(224), plt.imshow(
    imgBlur7, 'gray'), plt.title('7x7均值滤波'), plt.axis('off')
plt.show()
