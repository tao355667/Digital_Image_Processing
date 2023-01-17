import cv2 as cv  # opencv读取的格式是BGR
import sys  # exit()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# 创建和显示窗口
# nameWindow()
# imshow()
# destroyALLWindows()
# resizeWindow()

# cv.IMREAD_COLOR： 加载彩色图像。任何图像的透明度都会被忽视。它是默认标志。
# cv.IMREAD_GRAYSCALE：以灰度模式加载图像
# cv.IMREAD_UNCHANGED：加载图像，包括alpha通道
# 注意 除了这三个标志，你可以分别简单地传递整数1、0或-1。
img = cv.imread("basic/picture/cat.png", 1)
cv.namedWindow('Display window', cv.WINDOW_NORMAL)  # 创建窗口
if img is None:  # 没读到图 为空
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)  # 在窗口中显示图
k = cv.waitKey(0)  # 等待一个按键
if k == ord("s"):  # 按键为s则执行操作
    cv.imwrite("starry_night.png", img)  # 存储图像
cv.destroyAllWindows()  # 释放窗口资源
