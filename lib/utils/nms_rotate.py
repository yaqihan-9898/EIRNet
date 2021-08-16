# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import lib.config.config as cfg
import tensorflow as tf
# from lib.utils.rotate_polygon_nms import rotate_gpu_nms


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_angle_condition=False, angle_threshold=0, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,
                              angle_gap_threshold=angle_threshold,
                              use_angle_condition=use_angle_condition,
                              device_id=gpu_id)

        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfg.FLAGS.EPSILON)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)

def soft_nms_rotate_cpu(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
        py_cpu_softnms
        :param dets:   format [x, y, w, h,theta]
        :param sc:     score
        :param Nt:     iou threshold
        :param sigma:
        :param thresh: score threshold
        :param method:
        :return:      index to keep
        """
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    # y1 = dets[:, 0]
    # x1 = dets[:, 1]
    # y2 = dets[:, 2]
    # x2 = dets[:, 3]
    scores = sc
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = dets[i, 2] * dets[i, 3]  # N,1
    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        # xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        # yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        # xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        # yy2 = np.minimum(dets[i, 2], dets[pos:, 2])
        #
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = w * h
        # ovr = inter / (areas[i] + areas[pos:] - inter)
        iou =[]
        r1= ((dets[i, 0], dets[i, 1]), (dets[i, 2], dets[i, 3]), dets[i, 4])
        for j in range(pos, N):
            r2 = ((dets[j, 0], dets[j, 1]), (dets[j, 2], dets[j, 3]), dets[j, 4])
            # try:
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                inter = int_area * 1.0 / (r1 + r2 - int_area)
                iou.append(inter)
            else:
                iou.append(0.0)
            # except:
            #     """
            #       cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
            #       error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
            #     """
            #     inter = 0.9999
        ovr = np.array(iou, dtype=np.float32)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep

# def nms_rotate_gpu(boxes_list, scores, iou_threshold, use_angle_condition=False, angle_gap_threshold=0, device_id=0):
#     if use_angle_condition:
#         x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
#         boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
#         det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
#         keep = tf.py_func(rotate_gpu_nms,
#                           inp=[det_tensor, iou_threshold, device_id],
#                           Tout=tf.int64)
#         return keep
#     else:
#         x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
#         boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
#         det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
#         keep = tf.py_func(rotate_gpu_nms,
#                           inp=[det_tensor, iou_threshold, device_id],
#                           Tout=tf.int64)
#         keep = tf.reshape(keep, [-1])
#         return keep


if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    scores = np.array([0.99, 0.88, 0.66, 0.77])

    keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
                      0.7, 5)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with tf.Session() as sess:
        print(sess.run(keep))
