import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# 预处理
# %% 读图
img = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/urban_2.png', cv.IMREAD_GRAYSCALE)
equ = cv.equalizeHist(img)  # 均衡化
plt.figure('直方图均衡化')
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('ORIGINAL_GRAY')
plt.subplot(222), plt.hist(img.ravel(), 256, [
    0, 256]), plt.title('ORIGINAL_HIST')
plt.subplot(223), plt.imshow(equ, 'gray'), plt.title('EQU')
plt.subplot(224), plt.hist(equ.ravel(), 256, [0, 256]), plt.title('EQU_HIST')
plt.show()
# %% 模糊（平滑）
imgBlur1 = cv.GaussianBlur(img, (3, 3), 1)  # 高斯滤波
# canny边缘检测
imgCanny = cv.Canny(img, 0, 255)
imgBlur2 = cv.addWeighted(img, 0.5, imgCanny, 0.5, 0)
imgBlur3 = cv.blur(img, (3, 3))  # cv.blur 方法（均值滤波）比原图稍好
imgBlur4 = cv.equalizeHist(img)  # 均衡化-不如原图
plt.figure('图像平滑')
plt.subplot(221), plt.imshow(imgBlur1, 'gray'), plt.title('Gaussian')
plt.subplot(222), plt.imshow(imgBlur2, 'gray'), plt.title('Canny')
plt.subplot(223), plt.imshow(imgBlur3, 'gray'), plt.title('均值')
plt.subplot(224), plt.imshow(imgBlur4, 'gray'), plt.title('均衡化')
plt.show()
# %%自适应局部阈值处理
binaryMean = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
binaryMean1 = cv.adaptiveThreshold(
    imgBlur3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
plt.figure('')
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('img')
plt.subplot(222), plt.imshow(
    binaryMean, 'gray'), plt.title('img Adaptive threshold')
plt.subplot(223), plt.imshow(imgBlur3, 'gray'), plt.title('Blur')
plt.subplot(224), plt.imshow(
    binaryMean1, 'gray'), plt.title('imgBlur Adaptive threshold')
plt.show()
# %%
# 图像腐蚀
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode3 = cv.erode(binaryMean, kernel=kernel)  # 图像腐蚀
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode5 = cv.erode(binaryMean, kernel=kernel)  # 图像腐蚀
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode7 = cv.erode(binaryMean, kernel=kernel)  # 图像腐蚀
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgErode9 = cv.erode(binaryMean, kernel=kernel)  # 图像腐蚀

plt.figure('腐蚀')
plt.subplot(221), plt.imshow(imgErode3, 'gray'), plt.title('3x3')
plt.subplot(222), plt.imshow(imgErode5, 'gray'), plt.title('5x5')
plt.subplot(223), plt.imshow(imgErode7, 'gray'), plt.title('7x7')
plt.subplot(224), plt.imshow(imgErode9, 'gray'), plt.title('9x9')
plt.show()
# %%
# 图像的开运算
kSize = (3, 3)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen3 = cv.morphologyEx(binaryMean, cv.MORPH_OPEN, kernel)
kSize = (5, 5)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen5 = cv.morphologyEx(binaryMean, cv.MORPH_OPEN, kernel)
kSize = (7, 7)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen7 = cv.morphologyEx(binaryMean, cv.MORPH_OPEN, kernel)
kSize = (9, 9)  # 卷积核的尺寸
kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
imgOpen9 = cv.morphologyEx(binaryMean, cv.MORPH_OPEN, kernel)
plt.figure('开运算')
plt.subplot(221), plt.imshow(imgOpen3, 'gray'), plt.title('3x3')
plt.subplot(222), plt.imshow(imgOpen5, 'gray'), plt.title('5x5')
plt.subplot(223), plt.imshow(imgOpen7, 'gray'), plt.title('7x7')
plt.subplot(224), plt.imshow(imgOpen9, 'gray'), plt.title('9x9')
plt.show()
