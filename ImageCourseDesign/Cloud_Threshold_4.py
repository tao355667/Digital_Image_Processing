# -*- coding: utf-8 -*-
# %% 主观，客观地分析云层图像分割效果
import cv2 as cv  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# matplotlib中文字体
font = {'family': 'SimHei', 'weight': 'bold', 'size': '16'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
# %%云层分割函数


def CloudDiv(imgGray):
    # 分割（预处理）
    imgBlur = cv.blur(imgGray, (3, 3))  # 均值滤波
    th1ret, th1 = cv.threshold(
        imgBlur, 0, 256, cv.THRESH_BINARY+cv.THRESH_OTSU)  # OTSU
    kSize = (3, 3)  # 卷积核的尺寸
    kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
    imgOpen3 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)  # 开运算
    return imgOpen3


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
    './CourseDesignData/Dataset_cloud/test_img/barren_13.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray2 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/barren_29.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray3 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/urban_2.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray4 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/vegetation_2.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray5 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/vegetation_22.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray6 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/water_2.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray7 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_img/water_10.png', cv.IMREAD_GRAYSCALE)  # 灰度图
imgGray = [imgGray1, imgGray2, imgGray3, imgGray4,
           imgGray5, imgGray6, imgGray7]
# 读labels图
imgL1 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/barren_13.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL2 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/barren_29.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL3 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/urban_2.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL4 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/vegetation_2.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL5 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/vegetation_22.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL6 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/water_2.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL7 = cv.imread(
    './CourseDesignData/Dataset_cloud/test_label/water_10.tif', cv.IMREAD_GRAYSCALE)  # 灰度图
imgL = [imgL1, imgL2, imgL3, imgL4,
        imgL5, imgL6, imgL7]
imgLB = []
for i in range(7):  # labels二值化
    _, thth = cv.threshold(imgL[i], 1, 255, cv.THRESH_BINARY)
    imgLB.append(thth)
# 计算分割结果
imgDiv1 = []  # 云层分割结果
for i in range(7):
    imgDiv1.append(CloudDiv(imgGray[i]))
plt.figure('分割结果对比')
for i in range(7):  # 分割结果
    plt.subplot(2, 7, 1+i), plt.imshow(
        imgDiv1[i], 'gray'), plt.title('云层分割结果'+str(i+1)), plt.axis('off')
for i in range(7):  # labels分割结果
    plt.subplot(2, 7, 8+i), plt.imshow(
        imgLB[i], 'gray'), plt.title('labels '+str(i+1)), plt.axis('off')
plt.show()

# %%结果的量化评价
PA = []
MIOU = []
for i in range(7):
    # 评价云层分割结果
    PA.append(pixel_accuracy(imgDiv1[i], imgLB[i]))
    MIOU.append(mean_IU(imgDiv1[i], imgLB[i]))
print('量化评价指标：')
print(PA)
print(MIOU)
