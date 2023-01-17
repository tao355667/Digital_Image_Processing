# -*- coding: utf-8 -*-
# 基于阈值的河流图像分割，对比不同预处理的效果
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
    './CourseDesignData/dataset-river/train/46_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB原始图
imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)   # RGB转灰度

# %% 灰度图预处理
imgEqu = cv.equalizeHist(imgGray)  # 均衡化
# %%分割-均衡化后#三种阈值处理，绘图
_, th1 = cv.threshold(imgEqu, 127, 255, cv.THRESH_BINARY)  # 单一阈值
th2ret, th2 = cv.threshold(
    imgEqu, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
th3 = cv.adaptiveThreshold(
    imgEqu, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理

plt.figure('阈值分割')
plt.subplot(221), plt.imshow(imgEqu, 'gray'), plt.title(
    '均衡化后的图像'), plt.axis('off')
plt.subplot(222), plt.imshow(th1, 'gray'), plt.title('单一阈值分割'), plt.axis('off')
plt.subplot(223), plt.imshow(th2, 'gray'), plt.axis(
    'off'), plt.title("OTSU(T={})".format(th2ret))
plt.subplot(224), plt.imshow(th3, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()
# %%分割-未均衡化
_, th11 = cv.threshold(imgGray, 127, 255, cv.THRESH_BINARY)  # 单一阈值
th2ret, th22 = cv.threshold(
    imgGray, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
th33 = cv.adaptiveThreshold(
    imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
plt.figure('阈值分割')
plt.subplot(221), plt.imshow(
    imgGray, 'gray'), plt.title('灰度图'), plt.axis('off')
plt.subplot(222), plt.imshow(th11, 'gray'), plt.title(
    '单一阈值分割'), plt.axis('off')
plt.subplot(223), plt.imshow(th22, 'gray'), plt.title(
    "OTSU(T={})".format(th2ret)), plt.axis('off')
plt.subplot(224), plt.imshow(th33, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()
# %%分割-均值滤波
imgBlur = cv.blur(imgGray, (3, 3))  # 均值滤波
_, th11 = cv.threshold(imgBlur, 127, 255, cv.THRESH_BINARY)  # 单一阈值
th2ret, th22 = cv.threshold(
    imgBlur, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
th33 = cv.adaptiveThreshold(
    imgBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
plt.figure('阈值分割')
plt.subplot(221), plt.imshow(
    imgBlur, 'gray'), plt.title('灰度图均值滤波(3x3)'), plt.axis('off')
plt.subplot(222), plt.imshow(th11, 'gray'), plt.title(
    '单一阈值分割'), plt.axis('off')
plt.subplot(223), plt.imshow(th22, 'gray'), plt.title(
    "OTSU(T={})".format(th2ret)), plt.axis('off')
plt.subplot(224), plt.imshow(th33, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()

# %%分割-均值滤波
imgBlur = cv.blur(imgGray, (5, 5))  # 均值滤波
_, th11 = cv.threshold(imgBlur, 127, 255, cv.THRESH_BINARY)  # 单一阈值
th2ret, th22 = cv.threshold(
    imgBlur, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
th33 = cv.adaptiveThreshold(
    imgBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
plt.figure('阈值分割')
plt.subplot(221), plt.imshow(
    imgBlur, 'gray'), plt.title('灰度图均值滤波(5x5)'), plt.axis('off')
plt.subplot(222), plt.imshow(th11, 'gray'), plt.title(
    '单一阈值分割'), plt.axis('off')
plt.subplot(223), plt.imshow(th22, 'gray'), plt.title(
    "OTSU(T={})".format(th2ret)), plt.axis('off')
plt.subplot(224), plt.imshow(th33, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()
# %%分割-均值滤波
imgBlur = cv.blur(imgGray, (7, 7))  # 均值滤波
_, th11 = cv.threshold(imgBlur, 127, 255, cv.THRESH_BINARY)  # 单一阈值
th2ret, th22 = cv.threshold(
    imgBlur, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
th33 = cv.adaptiveThreshold(
    imgBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
plt.figure('阈值分割')
plt.subplot(221), plt.imshow(
    imgBlur, 'gray'), plt.title('灰度图均值滤波(7x7)'), plt.axis('off')
plt.subplot(222), plt.imshow(th11, 'gray'), plt.title(
    '单一阈值分割'), plt.axis('off')
plt.subplot(223), plt.imshow(th22, 'gray'), plt.title(
    "OTSU(T={})".format(th2ret)), plt.axis('off')
plt.subplot(224), plt.imshow(th33, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()
