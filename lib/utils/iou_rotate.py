# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from lib.utils.bbox_transform import *
# from lib.utils.rbbox_overlaps import rbbx_overlaps
# from lib.utils.iou_cpu import get_iou_matrix
import numpy as np
from lib.config import config as cfg
#
# def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
#     '''
#
#     :param boxes_list1:[N, 8] tensor
#     :param boxes_list2: [M, 8] tensor
#     :return:
#     '''
#
#     boxes1 = tf.cast(boxes1, tf.float32)
#     boxes2 = tf.cast(boxes2, tf.float32)
#     if use_gpu:
#
#         iou_matrix = tf.py_func(rbbx_overlaps,
#                                 inp=[boxes1, boxes2, gpu_id],
#                                 Tout=tf.float32)
#     else:
#         iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
#                                 Tout=tf.float32)
#
#     iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
#
#     return iou_matrix
#
#
def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        # print(box1)
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)


    return np.array(ious, dtype=np.float32)


#
# def iou_rotate_calculate2(boxes1, boxes2):
#     ious = []
#     if boxes1.shape[0] != 0:
#         area1 = boxes1[:, 2] * boxes1[:, 3]
#         area2 = boxes2[:, 2] * boxes2[:, 3]
#         for i in range(boxes1.shape[0]):
#             temp_ious = []
#             r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
#             r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])
#
#             int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
#             if int_pts is not None:
#                 order_pts = cv2.convexHull(int_pts, returnPoints=True)
#
#                 int_area = cv2.contourArea(order_pts)
#
#                 inter = int_area * 1.0 / (area1[i] + area2[i] - int_area)
#                 temp_ious.append(inter)
#             else:
#                 temp_ious.append(0.0)
#             ious.append(temp_ious)
#
#     return np.array(ious, dtype=np.float32)

def iou_rotate_calculate3(boxes1, boxes2,fg_num,use_gpu=True, gpu_id=0):
    '''

    :param boxes1:
    :param boxes2:
    :param fg_num:
    :param use_gpu:
    :param gpu_id:
    :return:
    '''
    ious = []
    assert fg_num>0
    if fg_num !=0:
        for i in range(fg_num):
            box1 = boxes1[i]
            cls = int(boxes1[i][-1])
            # print(cls)
            # print(box1)
            box2 = boxes2[i, 5 * cls:5 * (cls+ 1)]  # 64,5
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            # print(area1)
            temp_ious = []
            r1 = ((box1[0],box1[1]),(box1[2],box1[3]),box1[4])
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1 + area2- int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.00001)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32),box2

def iou_rotate_calculate4(boxes1, boxes2,gt_boxes,fg_num,use_gpu=True, gpu_id=0):
    '''
    当前预测框和其他所有gt的iou
    :param boxes1:对应gt
    :param boxes2:预测框
    :param fg_num:
    :param use_gpu:
    :param gpu_id:
    :return:
    '''
    ious = []
    ious_gt=[]
    assert fg_num>0
    if fg_num !=0:
        for i in range(fg_num):
            box1 = boxes1[i]
            cls = int(boxes1[i][-1])
            # print(cls)
            # print(box1)
            box2 = boxes2[i, 5 * cls:5 * (cls+ 1)]  # 64,5
            area3 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            r3 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            temp_ious = []
            temp_ious_gt = []
            # print('gt',box1)
            # print(box1.shape)
            for it in gt_boxes:
                # print(it)
                # print(it.shape)
                if it[0] == box1[0]:
                    temp_ious.append(0.0)
                    temp_ious_gt.append(0.0)
                else:
                    area1 = it[2] * it[3]
                    r1 = ((it[0], it[1]), (it[2], it[3]), it[4])
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area1 + area2 - int_area)
                        temp_ious.append(inter)
                    else:
                        temp_ious.append(0.0)
                    int_pts = cv2.rotatedRectangleIntersection(r1, r3)[1]
                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area1 + area3 - int_area)
                        temp_ious_gt.append(inter)
                    else:
                        temp_ious_gt.append(0.0)

            ious.append(temp_ious)
            ious_gt.append(temp_ious_gt)
    ious = np.array(ious, dtype=np.float32)
    ious_gt = np.array(ious_gt, dtype=np.float32)

    ious =abs(ious - ious_gt)

    # print('**'*20)
    # print('iou_with_other',ious)
    # print(ious.shape)
    ind = ious > 0
    ind.astype(np.float32)
    ind = np.sum(ind,axis=1)
    # print(ind.shape)
    if ious.shape[0] !=0:
        iou_mean=np.sum(np.exp(ious)-1,axis=1)/(ind+cfg.FLAGS.EPSILON)
        # iou_mean = np.sum(-np.log(1-ious), axis=1) / (ind + cfg.FLAGS.EPSILON)

    else:
        iou_mean = np.zeros(fg_num)

    return iou_mean.astype(np.float32)



if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    # boxes1 = np.array([[50, 50, 100, 300, 0],
    #                    [60, 60, 100, 200, 0]], np.float32)
    #
    # boxes2 = np.array([[50, 50, 100, 300, -45.],
    #                    [200, 200, 100, 200, 0.]], np.float32)
    #
    # start = time.time()
    # with tf.Session() as sess:
    #     ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)
    #     print(sess.run(ious))
    #     print('{}s'.format(time.time() - start))



    # start = time.time()
    # for _ in range(10):
    #     ious = rbbox_overlaps.rbbx_overlaps(boxes1, boxes2)
    # print('{}s'.format(time.time() - start))
    # print(ious)

    # print(ovr)



