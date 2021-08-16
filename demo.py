#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yaqi Han, based on code from Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.datasets import pascal_voc
from lib.utils.test import im_detect
from lib.nets.resnet import resnet
from lib.utils.timer import Timer
import xml.etree.ElementTree as ET
from lib.utils import nms_rotate
from lib.utils.bbox_transform import forward_convert

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def demo(sess, net, image_name,times):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('./data/VOCdevkit2007/VOC2007/JPEGImages', image_name)
    im = cv2.imread(im_file)
    im_temp = im
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    scores, boxes_r = im_detect(sess, net, im)

    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes_r.shape[0]))
    times.append(timer.total_time)

    print(np.max(scores[:, 1:]))

    for cls_ind, cls in enumerate(pascal_voc.CLASSES[1:]):
        filename= get_path()
        filename= filename.format(cls)
        cls_ind += 1  # because we skipped background
        cls_boxes_r = boxes_r[:, 5 * cls_ind:5 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets_r = np.hstack((cls_boxes_r,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        inds_r = np.where(dets_r[:, -1] >= cfg.FLAGS.score_filter)[0]
        cls_boxes_r = cls_boxes_r[inds_r, :]
        cls_scores=cls_scores[inds_r]
        dets_r = dets_r[inds_r, :]


        keep_r = nms_rotate.nms_rotate_cpu(boxes=np.array(cls_boxes_r),
                                            scores=np.array(cls_scores),
                                            iou_threshold=cfg.FLAGS.RNMS_threshold,
                                            max_output_size=500)


        dets_r = dets_r[keep_r, :]
        img = image_name.split(".")

        if dets_r.shape[0] != 0:
            for i in range(dets_r.shape[0]):
                with open(filename, 'a') as f:
                    f.write(
                        '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img[0], dets_r[i, 5], dets_r[i, 0],
                                                                                  dets_r[i, 1],
                                                                                  dets_r[i, 2], dets_r[i, 3],
                                                                                  dets_r[i, 4]))


def get_path():
    filename = 'test_{:s}.txt'
    path = os.path.join(
        './output/dets_r/'+cfg.network,
        filename)
    return path


if __name__ == '__main__':
    tfmodel = tf.train.latest_checkpoint(cfg.get_output_dir())
    print(tfmodel)

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if cfg.FLAGS.backbone.startswith('resnet'):
        net = resnet(batch_size=cfg.FLAGS.ims_per_batch)
    else:
        raise NotImplementedError


    n_classes = len(pascal_voc.CLASSES)

    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default')
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    fi = open('./data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt')
    txt = fi.readlines()
    im_names = []
    for line in txt:
        line = line.strip('\n')
        line = (line + cfg.FLAGS.image_ext)
        im_names.append(line)
    fi.close()

    for cls_ind, cls in enumerate(pascal_voc.CLASSES[1:]):
        filename= get_path()
        if not os.path.exists(os.path.join('./output/dets_r/'+cfg.network)):
            os.makedirs(os.path.join('./output/dets_r/'+cfg.network))
        filename = filename.format(cls)
        file = open(filename, 'w')
        file.close

    times=[]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name,times)
    print(np.average(np.array(times)))


