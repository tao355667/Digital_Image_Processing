# -*- coding: utf-8 -*-
# %% 形态学处理（方法2，反相）
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
# %%分割（不预处理）
th1 = cv.adaptiveThreshold(
    imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
plt.figure('阈值分割，形态学')
plt.subplot(111), plt.imshow(th1, 'gray'), plt.title(
    '自适应局部阈值处理'), plt.axis('off')
plt.show()
# %%图像腐蚀
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode3 = cv.erode(th1, kernel=kernel)  # 图像膨胀
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode5 = cv.erode(th1, kernel=kernel)
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode7 = cv.erode(th1, kernel=kernel)
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode9 = cv.erode(th1, kernel=kernel)

plt.figure('腐蚀')
plt.subplot(221), plt.imshow(
    imgErode3, 'gray'), plt.title('腐蚀3x3'), plt.axis('off')
plt.subplot(222), plt.imshow(
    imgErode5, 'gray'), plt.title('腐蚀5x5'), plt.axis('off')
plt.subplot(223), plt.imshow(
    imgErode7, 'gray'), plt.title('腐蚀7x7'), plt.axis('off')
plt.subplot(224), plt.imshow(
    imgErode9, 'gray'), plt.title('腐蚀9x9'), plt.axis('off')
plt.show()
# %% 图像的开运算
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen3 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen5 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen7 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen9 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
plt.figure('开运算')
plt.subplot(221), plt.imshow(imgOpen3, 'gray'), plt.title(
    '开运算3x3'), plt.axis('off')
plt.subplot(222), plt.imshow(imgOpen5, 'gray'), plt.title(
    '开运算5x5'), plt.axis('off')
plt.subplot(223), plt.imshow(imgOpen7, 'gray'), plt.title(
    '开运算7x7'), plt.axis('off')
plt.subplot(224), plt.imshow(imgOpen9, 'gray'), plt.title(
    '开运算9x9'), plt.axis('off')
plt.show()

# %% 图像膨胀
rth1 = 255-th1
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgDilate3 = cv.dilate(rth1, kernel=kernel)  # 图像膨胀
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgDilate5 = cv.dilate(rth1, kernel=kernel)
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgDilate7 = cv.dilate(rth1, kernel=kernel)
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgDilate9 = cv.dilate(rth1, kernel=kernel)

plt.figure('膨胀')
plt.subplot(221), plt.imshow(
    imgDilate3, 'gray'), plt.title('膨胀3x3'), plt.axis('off')
plt.subplot(222), plt.imshow(
    imgDilate5, 'gray'), plt.title('膨胀5x5'), plt.axis('off')
plt.subplot(223), plt.imshow(
    imgDilate7, 'gray'), plt.title('膨胀7x7'), plt.axis('off')
plt.subplot(224), plt.imshow(
    imgDilate9, 'gray'), plt.title('膨胀9x9'), plt.axis('off')
plt.show()

# %% 图像的闭运算
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgClose3 = cv.morphologyEx(rth1, cv.MORPH_CLOSE, kernel)
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgClose5 = cv.morphologyEx(rth1, cv.MORPH_CLOSE, kernel)
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgClose7 = cv.morphologyEx(rth1, cv.MORPH_CLOSE, kernel)
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgClose9 = cv.morphologyEx(rth1, cv.MORPH_CLOSE, kernel)
plt.figure('开运算')
plt.subplot(221), plt.imshow(imgClose3, 'gray'), plt.title(
    '闭运算3x3'), plt.axis('off')
plt.subplot(222), plt.imshow(imgClose5, 'gray'), plt.title(
    '闭运算5x5'), plt.axis('off')
plt.subplot(223), plt.imshow(imgClose7, 'gray'), plt.title(
    '闭运算7x7'), plt.axis('off')
plt.subplot(224), plt.imshow(imgClose9, 'gray'), plt.title(
    '闭运算9x9'), plt.axis('off')
plt.show()

# %% 找连通域并剔除面积小的连通域
# 去除小白点
rxtimg = imgClose5  # 找效果较好的形态学图，之后会对该图进行处理
num_labels2, labels2, stats2, centroids2 = cv.connectedComponentsWithStats(
    rxtimg, connectivity=8)
bigp2 = []  # 大面积区域下标
smallp2 = []  # 小面积区域下标
A = 850  # 面积阈值
for i in range(num_labels2):
    if (stats2[i][4] >= A):  # 面积筛选
        bigp2.append(i)
    else:
        smallp2.append(i)
imgf2 = np.zeros([rxtimg.shape[0], rxtimg.shape[1]])
for x in range(imgf2.shape[0]):  # 将大面积区域绘制在新图上
    for y in range(imgf2.shape[1]):
        if labels2[x][y] in bigp2:
            if labels2[x][y]:
                imgf2[x][y] = 255
        else:
            pass  # imgf2[x][y] = 0

# 去除小黑点（反相），处理结束不需反相
finalimg = 255-imgf2.astype(np.uint8)
num_labels3, labels3, stats3, centroids3 = cv.connectedComponentsWithStats(
    finalimg, connectivity=8)
bigp3 = []  # 大面积区域下标
smallp3 = []  # 小面积区域下标
A = 850
for i in range(num_labels3):
    if (stats3[i][4] >= A):  # 面积筛选
        bigp3.append(i)
    else:
        smallp3.append(i)
imgf3 = np.zeros([finalimg.shape[0], finalimg.shape[1]])
for x in range(imgf3.shape[0]):  # 将大面积区域绘制在新图上
    for y in range(imgf3.shape[1]):
        if labels3[x][y] in bigp3:
            if labels3[x][y]:
                imgf3[x][y] = 255
        else:
            pass  # imgf2[x][y] = 0

plt.figure('连通域')
plt.subplot(211), plt.imshow(imgf2, 'gray'), plt.title(
    '反向图闭运算后，面积筛选'), plt.axis('off')
plt.subplot(212), plt.imshow(imgf3, 'gray'), plt.title(
    '去除小黑点（反相），处理结束不需反相'), plt.axis('off')
plt.show()
