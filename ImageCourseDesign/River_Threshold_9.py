# -*- coding: utf-8 -*-
# 使用一个阈值区间进行分割
import cv2 as cv  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# matplotlib中文字体
font = {'family': 'SimHei', 'weight': 'bold', 'size': '10'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
# %% 读图，转灰度
img1 = cv.imread(
    './CourseDesignData/dataset-river/train/4_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img2 = cv.imread(
    './CourseDesignData/dataset-river/train/37_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img3 = cv.imread(
    './CourseDesignData/dataset-river/train/43_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img4 = cv.imread(
    './CourseDesignData/dataset-river/train/45_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img5 = cv.imread(
    './CourseDesignData/dataset-river/train/46_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img6 = cv.imread(
    './CourseDesignData/dataset-river/train/48_0_4461.png', cv.IMREAD_COLOR)  # BGR原始图
img = [img1, img2, img3, img4, img5, img6]
imgGray = []
for i in np.arange(6):
    img[i] = cv.cvtColor(img[i], cv.COLOR_BGR2RGB)  # RGB原始图
    imgGray.append(cv.cvtColor(img[i], cv.COLOR_RGB2GRAY))   # RGB转灰度
# 原始图+灰度图
plt.figure('原始图+灰度图')
for i in np.arange(6):
    plt.subplot(
        3, 4, 2*i+1), plt.imshow(img[i]), plt.title('原始图'+str(i)), plt.axis('off')
    plt.subplot(
        3, 4, 2*i+2), plt.imshow(imgGray[i], 'gray'), plt.title('灰度图'+str(i)), plt.axis('off')
plt.show()
# %% 观察灰度直方图
plt.figure('观察灰度直方图')
for i in np.arange(6):
    plt.subplot(
        3, 4, 2*i+1), plt.imshow(imgGray[i], 'gray'), plt.title('灰度图'+str(i)), plt.axis('off')
    plt.subplot(
        3, 4, 2*i+2), plt.hist(imgGray[i].ravel(), 256, [0, 256]), plt.title('灰度直方图'+str(i))
plt.show()
# %% 灰度图预处理
# imgBlur = cv.blur(imgGray[0], (3, 3))  # 均值滤波
# imgEqu = cv.equalizeHist(imgBlur)  # 均衡化
# imgEqu = 255-imgEqu
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized = clahe.apply(imgGray[0])
# # 原始图+灰度图
# plt.figure('直方图均衡化')
# plt.subplot(221), plt.imshow(imgGray, 'gray'), plt.title('灰度图')
# plt.subplot(222), plt.hist(imgGray.ravel(), 256, [
#     0, 256]), plt.title('灰度直方图')
# plt.subplot(223), plt.imshow(imgEqu, 'gray'), plt.title('均衡化')
# plt.subplot(224), plt.hist(imgEqu.ravel(), 256,
#                            [0, 256]), plt.title('均衡化后灰度直方图')
# plt.show()
