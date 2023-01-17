# -*- coding: utf-8 -*-
# %% 主观，客观地分析河流图像分割效果
import cv2 as cv  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# matplotlib中文字体
font = {'family': 'SimHei', 'weight': 'bold', 'size': '16'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
# %%河流分割函数


def RiverDiv1(imgGray):
    # 分割（不预处理）
    th1 = cv.adaptiveThreshold(
        imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
    # 图像的开运算
    kSize = (5, 5)  # 卷积核的尺寸
    kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
    imgOpen5 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
    # 找连通域并剔除面积小的连通域
    xtimg = imgOpen5  # 找效果较好的形态学图，之后会对该图进行处理
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        xtimg, connectivity=8)  # 提取连通域
    # 处理图中小白点
    bigp = []  # 大面积区域下标
    smallp = []  # 小面积区域下标
    A = 850  # 面积阈值
    for i in range(num_labels):
        if (stats[i][4] >= A):  # 面积筛选
            bigp.append(i)
        else:
            smallp.append(i)
    imgf1 = np.zeros([xtimg.shape[0], xtimg.shape[1]])
    for x in range(imgf1.shape[0]):  # 将大面积区域绘制在新图上
        for y in range(imgf1.shape[1]):
            if labels[x][y] in bigp:
                if labels[x][y]:
                    imgf1[x][y] = 255
            else:
                pass  # imgf1[x][y] = 0
    # 处理小黑点（反相处理，处理结束再反相）
    finalimg = 255-imgf1.astype(np.uint8)
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
    imgf3 = 255-imgf3  # 反相
    return imgf3


def RiverDiv2(imgGray):  # 第二种分割方式
    # 分割（不预处理）
    th1 = cv.adaptiveThreshold(
        imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)  # 自适应局部阈值处理
    # 闭运算
    rth1 = 255-th1  # 反向图
    kSize = (5, 5)  # 卷积核的尺寸
    kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
    imgClose5 = cv.morphologyEx(rth1, cv.MORPH_CLOSE, kernel)
    # 找连通域并剔除面积小的连通域
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
    return imgf3
# %%量化评价指标函数


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# %% 读图，转灰度
imgGray1 = cv.imread(
    './CourseDesignData/dataset-river/train/4_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray2 = cv.imread(
    './CourseDesignData/dataset-river/train/37_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray3 = cv.imread(
    './CourseDesignData/dataset-river/train/40_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray4 = cv.imread(
    './CourseDesignData/dataset-river/train/42_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray5 = cv.imread(
    './CourseDesignData/dataset-river/train/43_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray6 = cv.imread(
    './CourseDesignData/dataset-river/train/45_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray7 = cv.imread(
    './CourseDesignData/dataset-river/train/46_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray8 = cv.imread(
    './CourseDesignData/dataset-river/train/48_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray = [imgGray1, imgGray2, imgGray3, imgGray4,
           imgGray5, imgGray6, imgGray7, imgGray8]
# 读labels图
imgL1 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/4_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL2 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/37_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL3 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/40_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL4 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/42_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL5 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/43_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL6 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/45_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL7 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/46_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL8 = cv.imread(
    './CourseDesignData/dataset-river/train_labels/48_0_4461.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL = [imgL1, imgL2, imgL3, imgL4,
        imgL5, imgL6, imgL7, imgL8]
imgLB = []
for i in range(8):  # labels二值化
    _, thth = cv.threshold(imgL[i], 1, 255, cv.THRESH_BINARY)
    imgLB.append(thth)
# print(imgLB[0])
# for x in range(imgLB[0].shape[0]):
#     for y in range(imgLB[0].shape[1]):
#         print(imgLB[0][x][y],end='')
#     print()
# 计算分割结果
imgDiv1 = []  # 方式1分割结果
for i in range(8):
    imgDiv1.append(RiverDiv1(imgGray[i]))
imgDiv2 = []  # 方式2分割结果
for i in range(8):
    imgDiv2.append(RiverDiv2(imgGray[i]))
plt.figure('分割结果对比')
for i in range(8):  # 方式1分割结果
    plt.subplot(3, 8, 1+i), plt.imshow(
        imgDiv1[i], 'gray'), plt.title('方式1分割结果'+str(i+1)), plt.axis('off')
for i in range(8):  # 方式2分割结果
    plt.subplot(3, 8, 9+i), plt.imshow(
        imgDiv2[i], 'gray'), plt.title('方式2分割结果'+str(i+1)), plt.axis('off')
for i in range(8):  # labels分割结果
    plt.subplot(3, 8, 17+i), plt.imshow(
        imgLB[i], 'gray'), plt.title('labels '+str(i+1)), plt.axis('off')
plt.show()

# %%结果的量化评价
PA1 = []
MIOU1 = []
PA2 = []
MIOU2 = []
for i in range(8):
    # 评价方式1分割结果
    PA1.append(pixel_accuracy(imgDiv1[i], imgLB[i]))
    MIOU1.append(mean_IU(imgDiv1[i], imgLB[i]))
    # 评价方式2分割结果
    PA2.append(pixel_accuracy(imgDiv2[i], imgLB[i]))
    MIOU2.append(mean_IU(imgDiv2[i], imgLB[i]))
print('量化评价指标：')
print(PA1)
print(MIOU1)
print(PA2)
print(MIOU2)
