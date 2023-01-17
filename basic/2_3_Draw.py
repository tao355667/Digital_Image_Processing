# 要绘制一条线，您需要传递线的开始和结束坐标。我们将创
# 建一个黑色图像，并从左上角到右下角在其上绘制一条蓝线。
import numpy as np
import cv2 as cv
# 创建黑色的图像
img = np.zeros((512, 512, 3), np.uint8)
# 绘制一条厚度为5的蓝色对角线
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# Drawing Rectangle
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# Drawing Circle
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
# Drawing Ellipse
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
# Drawing Polygon
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))
# Adding Text to Images:
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)
# 显示图像
cv.imshow("Display window", img)
k = cv.waitKey(0)  # 等待一个按键
cv.destroyAllWindows()  # 释放窗口资源
