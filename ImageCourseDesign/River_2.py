import numpy as np
import cv2 as cv


# name是打开的图像窗口名，img是图像
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Trackbar回调函数


def nothing(x):
    pass


# 读图
img = cv.imread(
    './CourseDesignData/dataset-river/train/16_0_4461.png', cv.IMREAD_GRAYSCALE)

cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('m', 'image', 0, 255, nothing)
while (1):
    k = cv.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    # get current positions of trackbar
    m = cv.getTrackbarPos('m', 'image')
    ret, thresh1 = cv.threshold(img, m, 255, cv.THRESH_BINARY)
    cv.imshow('image', thresh1)
cv.destroyAllWindows()
