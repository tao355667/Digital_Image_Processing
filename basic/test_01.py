import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = cv2.imread('basic/picture/Lena.jpg')

img  # 显示图像（用数组的形式）

# 图像的显示，也可以创建多个窗口
cv2.imshow('image', img)
# 等待时间，毫秒级，0表示任意键终止
cv2.waitKey(0)
cv2.destroyAllWindows()

# 将上面步骤定义为一个函数
# name是打开的图像窗口名，img是图像


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img.shape  # [高h][宽w][通道数]
img = cv2.imread('basic/picture/Lena.jpg', cv2.IMREAD_GRAYSCALE)
img
img.shape
# 图像的显示，也可以创建多个窗口
cv2.imshow('image', img)
# 等待时间，毫秒级，0表示任意键终止
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 保存图片
cv2.imwrite('basic/picture/mycat.png', img)
type(img)  # 图像的数据类型
vc = cv2.VideoCapture('basic/video/video01.mp4')
# 读取视频
# 检查是否打开正确
if vc.isOpened():  # vc能打开
    open, frame = vc.read()  # 函数有两个返回值，open表示读取是否成功，frame是读取的图像
else:
    open = False
while open:
    ret, frame = vc.read()
    if frame is None:  # 读到的图为空
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图片转换为灰度图
        cv2.imshow('result', gray)  # 显示转换后的灰度图
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
# %%
img = cv2.imread('basic/picture/Lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR图转化为RGB，因为plt.imshow用的是RGB
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)  # 上下左右填充大小均为50
# 复制边缘的像素点
replicate = cv2.copyMakeBorder(
    img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# 反射，以图像边缘为轴对称,gfedcba|abcdefg|afedcba
refliect = cv2.copyMakeBorder(
    img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
# 反射，以最边缘像素为轴对称，gfedcb|abcdefg|fedcba
refliect101 = cv2.copyMakeBorder(
    img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# 外包装法，cdefg|abcdefgh|abcdefg
wrap = cv2.copyMakeBorder(img, top_size, bottom_size,
                          left_size, right_size, cv2.BORDER_WRAP)
# 常量法，常数值填充
constant = cv2.copyMakeBorder(
    img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(refliect, 'gray'), plt.title('REFLIECT')
plt.subplot(234), plt.imshow(refliect101, 'gray'), plt.title('REFLIECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()
# %%
img = cv2.imread('basic/picture/Lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转换为灰度图
# cv2.imwrite('picture/mygray.jpg',img_gray)#存储灰度图
# 5种阈值处理方式
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV',
          'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


img = cv2.imread('basic/picture/Lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR图转化为RGB，因为plt.imshow用的是RGB
plt.imshow(img, 'gray')
plt.show()
# %%
