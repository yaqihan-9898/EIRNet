# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.proposal_top_layer import proposal_top_layer
from lib.layer_utils.uplevel_utils import uplevel_utils
from lib.layer_utils.get_mask import get_mask,get_mask_plus
from lib.layer_utils.cls_weight import get_weight
from lib.utils.noise import add_gaussian_noise,short_side_resize
from lib.datasets.flip_v import flip
# from lib.utils.coordinate_convert import back_forward_convert
from lib.utils.bbox_transform import bbox_transform_inv_r,back_forward_convert
from lib.utils.iou_rotate import iou_rotate_calculate3,iou_rotate_calculate4

class Network(object):
    def __init__(self, batch_size=1):
        self._feat_stride = [8 ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._channels = 1024
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}
        self._labels=[]
        self._uplabels=[]
        self._fpn_anchor_scales=[4, 8, 16,32] # 64,128,256,512
        self.anchor_list=[]

    # Summaries #
    def _add_image_summary(self, image, boxes):
        # add back mean
        image += cfg.FLAGS2["pixel_means"]
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # dims for normalization
        # width = tf.to_float(tf.shape(image)[2])
        # height = tf.to_float(tf.shape(image)[1])
        # # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        # cols = tf.unstack(boxes, axis=1)
        # boxes = tf.stack([cols[1] / height,
        #                   cols[0] / width,
        #                   cols[3] / height,
        #                   cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        # boxes = tf.expand_dims(boxes, dim=0)  #(1, ?, 4)
        # image = tf.image.draw_bounding_boxes(image, boxes)
        # boxes=self._add_mask_summary(self.attention,self.mask_boxes)
        # image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_mask_summary(self, image, boxes):
        # add back meanimage += cfg.FLAGS2["pixel_means"]
        # bgr to rgb (opencv uses bgr)
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]

        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[0] / height,
                          cols[1] / width,
                          cols[2] / height,
                          cols[3] / width], axis=1)

        # add batch dimension (assume batch_size==1)
        #assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        # image = tf.image.draw_bounding_boxes(image, boxes)
        return boxes
        # return tf.summary.image('mask_region', image)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.FLAGS.rpn_top_n, 5])
            rpn_scores.set_shape([cfg.FLAGS.rpn_top_n, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):

            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

            return rois, rpn_scores

    def _crop_pool_layer(self, bottom, rois, name):
        #roi pooling
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)


    def _proposal_target_layer(self, rois, roi_scores,rois_from_att, name):
        with tf.variable_scope(name):
            self._ro_gt_boxes_5 = tf.py_func(back_forward_convert,
                                       inp=[self._ro_gt_boxes],
                                       Tout=tf.float32)

            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights,bbox_targets_r, bbox_inside_weights_r, bbox_outside_weights_r,bbox_target_data_ro, fg_gt,fg_num = tf.py_func(
                    proposal_target_layer,
                    [rois, roi_scores, self._gt_boxes,self._ro_gt_boxes_5,self._num_classes,rois_from_att],
                    [tf.float32,tf.float32,tf.float32,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32])

            rois.set_shape([cfg.FLAGS.batch_size, 5])
            roi_scores.set_shape([cfg.FLAGS.batch_size])
            labels.set_shape([cfg.FLAGS.batch_size, 1])
            bbox_targets.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_targets_r.set_shape([cfg.FLAGS.batch_size, self._num_classes * 5])
            bbox_inside_weights_r.set_shape([cfg.FLAGS.batch_size, self._num_classes * 5])
            bbox_outside_weights_r.set_shape([cfg.FLAGS.batch_size, self._num_classes * 5])
            fg_gt = tf.reshape(fg_gt, [fg_num, 6])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['fg_gt']= fg_gt
            self._proposal_targets['fg_num'] = fg_num
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            self._proposal_targets['bbox_targets_r'] = bbox_targets_r
            self._proposal_targets['bbox_inside_weights_r'] = bbox_inside_weights_r
            self._proposal_targets['bbox_outside_weights_r'] = bbox_outside_weights_r

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores


    def build_network(self, sess, is_training=True):
        raise NotImplementedError

    def _smooth_l1_loss_base(self,bbox_pred, bbox_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        abs_box_diff = tf.abs(box_diff)

        smoothL1_sign = tf.stop_gradient(
            tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
        loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                   + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
        return loss_box
    def _smooth_l1_loss_fpn(self,bbox_pred, bbox_targets, label, sigma=1.0):
        value = self._smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
        value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
        rpn_positive = tf.where(tf.greater(label, 0))

        # rpn_select = tf.stop_gradient(rpn_select) # to avoid
        selected_value = tf.gather(value, rpn_positive)
        non_ignored_mask = tf.stop_gradient(
            1.0 - tf.to_float(tf.equal(label, -1)))  # positve is 1.0 others is 0.0

        bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))
        self._judge=tf.py_func(assert_loss_too_high,
                   [bbox_loss],[tf.float32])
        return bbox_loss
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _smooth_l1_loss_rcnn(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))

        im_size_min = tf.reduce_min(self._im_info[0,0:2])
        im_size_max = tf.reduce_max(self._im_info[0,0:2])
        im_scale = tf.to_float(cfg.FLAGS2["test_scales"]) / tf.to_float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        sign = tf.stop_gradient(tf.to_float(tf.less(tf.to_float(cfg.FLAGS.max_size), tf.to_float(tf.round(im_scale * im_size_max)))))
        im_scale = im_scale + sign * (tf.to_float(cfg.FLAGS.test_max_size) / tf.to_float(im_size_max)-im_scale)
        # self._proposal_targets['bbox_target_with_label']
        boxes = self._proposal_targets['rois'][:, 1:5] / im_scale
        stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds_ro"]), (self._num_classes))
        means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means_ro"]), (self._num_classes))
        bbox_pred *= stds
        bbox_pred += means
        # bbox_pred = tf.Print(bbox_pred, [tf.shape(bbox_pred)], summarize=10, message='bbox_pred ')
        boxes_pred = tf.py_func(bbox_transform_inv_r,[boxes,bbox_pred],[tf.float32])
        boxes_pred = tf.squeeze(boxes_pred)
        # boxes_pred = tf.Print(boxes_pred, [tf.shape(boxes_pred)], summarize=10, message='boxes_pred ')
        overlaps,box2 = tf.py_func(iou_rotate_calculate3,
                              inp=[ self._proposal_targets['fg_gt'],boxes_pred,self._proposal_targets['fg_num']],
                              Tout=[tf.float32,tf.float32])
        overlaps_with_other_box = tf.py_func(iou_rotate_calculate4,
                                    inp=[self._proposal_targets['fg_gt'], boxes_pred,self._ro_gt_boxes_5, self._proposal_targets['fg_num']],
                                    Tout=[tf.float32])
        self._show_pred_box = box2
        overlaps = tf.reshape(overlaps, [-1, 1])
        overlaps_with_other_box=tf.reshape(overlaps_with_other_box,[-1,1])

        iou_factor = tf.stop_gradient(tf.reduce_mean(-1 * tf.log(overlaps+ cfg.FLAGS.EPSILON)+1.5*overlaps_with_other_box)) / (
                tf.stop_gradient(loss_box) + cfg.FLAGS.EPSILON)

        return loss_box * iou_factor

    def build_attention_loss(self,mask, featuremap):
        shape = tf.shape(mask)
        featuremap = tf.image.resize_bilinear(featuremap, [shape[1], shape[2]])

        mask = tf.cast(mask, tf.int32)
        mask = tf.reshape(mask, [-1])

        featuremap = tf.reshape(featuremap, [-1, 2])
        attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask, logits=featuremap)
        attention_loss = tf.reduce_mean(attention_loss)
        return attention_loss


    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):

            # RPN, class loss
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_cls_score = self._predictions["rpn_cls_score"]
            rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_label, -1)), [-1])
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_labels = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                              labels=rpn_labels))

            # RPN, bbox loss
            rpn_box_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_labels = self._anchor_targets['rpn_labels']
            rpn_loss_box = self._smooth_l1_loss_fpn(bbox_pred=rpn_box_pred,
                                                    bbox_targets=rpn_bbox_targets,
                                                    label=rpn_labels,
                                                    sigma=sigma_rpn)



            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            self.show=label
            cls_weight=tf.py_func(get_weight, [label], [tf.float32])
            cross_entropy = tf.reduce_mean(  cls_weight*
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            # RCNN, rotated bbox loss
            if cfg.FLAGS.pre_ro == True:
                bbox_pred_r = self._predictions['ro_bbox_pred']
                bbox_targets_r = self._proposal_targets['bbox_targets_r']
                bbox_inside_weights_r = self._proposal_targets['bbox_inside_weights_r']
                bbox_outside_weights_r = self._proposal_targets['bbox_outside_weights_r']

                loss_box_r = self._smooth_l1_loss_rcnn(bbox_pred_r, bbox_targets_r, bbox_inside_weights_r,
                                                  bbox_outside_weights_r)
                self._losses['loss_box_r'] = loss_box_r
                tf.summary.scalar('loss_box_r', loss_box_r)
            else:
                self._losses['loss_box_r'] = tf.constant(0)
                tf.summary.scalar('loss_box_r', 0)

            # attention loss
            if cfg.FLAGS.attention:
                attention_loss = self.build_attention_loss(self._mask, self._predictions["net_attention"])
                attention_loss_plus = self.build_attention_loss(self._mask_plus, self._predictions["net_attention_plus"])
                self._losses['attention_loss'] = attention_loss+attention_loss_plus
                tf.summary.scalar('attention_loss', attention_loss+attention_loss_plus)
            else:
                self._losses['attention_loss'] = tf.constant(0)
                tf.summary.scalar('attention_loss', 0)


            # uplevel RCNN, class loss
            if cfg.FLAGS.use_uplevel_labels == True:
                uplevel_labels = tf.py_func(uplevel_utils, [label], [tf.int32])
                uplevel_labels = tf.reshape(uplevel_labels, [-1])

                # uplevel_cls_score = tf.py_func(convert_uplevel, [self._predictions["cls_score"]], [tf.float32])

                uplevel_cls_score = self._predictions["uplevel_cls_score"]  # [64，up_level_clc_num]

                # cls_weight_up = tf.py_func(get_weight_up, [uplevel_labels], [tf.float32])
                uplevel_cross_entropy = tf.reduce_mean(  #cls_weight_up*
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.reshape(uplevel_cls_score, [-1, cfg.FLAGS.uplevel_len]), labels=uplevel_labels))

                self._losses['uplevel_cls'] = uplevel_cross_entropy
                tf.summary.scalar('uplevel_cls', uplevel_cross_entropy)
            else:
                self._losses['uplevel_cls'] = tf.constant(0)
                tf.summary.scalar('uplevel_cls', 0)


            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            tf.summary.scalar('loss_cls', cross_entropy)
            tf.summary.scalar('loss_box', loss_box)
            tf.summary.scalar('rpn_cls', rpn_cross_entropy)
            tf.summary.scalar('rpn_loss_box', rpn_loss_box)
            if cfg.FLAGS.use_uplevel_labels == True:
                if cfg.FLAGS.double_cls:
                    sign = tf.stop_gradient(tf.to_float(tf.less(self.iter, cfg.FLAGS.double_iter)))
                    cross_entropy=2.0*cross_entropy+(1-sign)*2.0*cross_entropy
                    uplevel_cross_entropy=2.0 * uplevel_cross_entropy
                sign = tf.stop_gradient(tf.to_float(tf.less(self.iter, 500)))
                # k= sign*0.8+(1-sign)*1.0
                loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + uplevel_cross_entropy
                # loss = cross_entropy + loss_box + k*rpn_cross_entropy + k*rpn_loss_box + 1.5*uplevel_cross_entropy
            else:
                loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            if cfg.FLAGS.attention == True:
                loss = loss + self._losses['attention_loss']
            if cfg.FLAGS.pre_ro == True:
                loss = loss + self._losses['loss_box_r']
            else:
                self._losses['loss_box_r'] = tf.convert_to_tensor(0)
            self._losses['total_loss'] = loss
            tf.summary.scalar('total_loss', loss)
            self._event_summaries.update(self._losses)

        return loss

    '''anchor_scales=(8, 16, 32)'''
    def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(2,4, 8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._ro_gt_boxes = tf.placeholder(tf.float32, shape=[None, 9])
        self._tag = tag
        # 网络参数
        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = cfg.ANCHOR_NUMS

        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        if training:
            self.iter = tf.placeholder(tf.float32, shape=[None])
        else:
            self.iter=1
        assert tag != None
        # mask = self._get_mask(tf.squeeze(self._image, 0),self._ro_gt_boxes)
        self._mask = tf.py_func(
            get_mask,
            [tf.squeeze(self._image, 0),self._ro_gt_boxes],
            [tf.float32])
        self._mask_plus = tf.py_func(
            get_mask_plus,
            [tf.squeeze(self._image, 0), self._ro_gt_boxes],
            [tf.float32])
        tf.summary.image('mask_000',self._mask)
        tf.summary.image('mask_111', self._mask_plus)
        # handle most of the regularizer here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self.build_network(sess, training)

        layers_to_output = {'rois': rois}
        layers_to_output.update(self._predictions)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds"]), (self._num_classes))
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means"]), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means

            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds_ro"]), (self._num_classes))
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means_ro"]), (self._num_classes))
            self._predictions["ro_bbox_pred"] *= stds
            self._predictions["ro_bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        cls_score, cls_prob, ro_bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                                       self._predictions['cls_prob'],
                                                                       self._predictions["ro_bbox_pred"],
                                                                       self._predictions['rois']],
                                                                      feed_dict=feed_dict)
        return cls_score, cls_prob, ro_bbox_pred, rois




    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs,iter, train_op):
        img, h, r = flip(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['ro_gt_boxes'])

        iter = np.array([iter])
        feed_dict = {self._image: img, self._im_info: blobs['im_info'],
                     self._gt_boxes: h, self._ro_gt_boxes: r, self.iter: iter}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box, j, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
             self._losses['rpn_loss_box'],
             self._losses['cross_entropy'],
             self._losses['loss_box'],
             self._losses['loss_box_r'],
             self._losses['uplevel_cls'],
             self._losses['attention_loss'],
             self._losses['total_loss'],
             self._show_pred_box,
             self._judge,
             train_op],
            feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box

        # if cfg.FLAGS.use_uplevel_labels == True:
        #     if cfg.FLAGS.attention:
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, uplevel_cls, attention_loss,loss, pred_box,j, _ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #              self._losses['rpn_loss_box'],
        #              self._losses['cross_entropy'],
        #              self._losses['loss_box'],
        #              self._losses['loss_box_r'],
        #              self._losses['uplevel_cls'],
        #              self._losses['attention_loss'],
        #              self._losses['total_loss'],
        #              self._show_pred_box,
        #              self._judge,
        #              train_op],
        #             feed_dict=feed_dict)
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, uplevel_cls, attention_loss,loss,pred_box
        #     else:
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, uplevel_cls, loss, pred_box, show,_ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #              self._losses['rpn_loss_box'],
        #              self._losses['cross_entropy'],
        #              self._losses['loss_box'],
        #              self._losses['loss_box_r'],
        #              self._losses['uplevel_cls'],
        #              self._losses['total_loss'],
        #              self._show_pred_box,
        #              self.show,
        #              train_op],
        #             feed_dict=feed_dict)
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, uplevel_cls, loss
        #
        # else:
        #     if cfg.FLAGS.attention:
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, attention_loss,loss, _ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #              self._losses['rpn_loss_box'],
        #              self._losses['cross_entropy'],
        #              self._losses['loss_box'],
        #              self._losses['loss_box_r'],
        #              self._losses['attention_loss'],
        #              self._losses['total_loss'],
        #              train_op],
        #             feed_dict=feed_dict)
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, attention_loss, loss
        #     else:
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, loss,box, _ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #              self._losses['rpn_loss_box'],
        #              self._losses['cross_entropy'],
        #              self._losses['loss_box'],
        #              self._losses['loss_box_r'],
        #              self._losses['total_loss'],
        #              self._predictions['bbox_pred'],
        #              train_op],
        #             feed_dict=feed_dict)
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,loss_box_r, loss


    def train_step_with_summary(self, sess, blobs, iter, train_op):
        iter = np.array([iter])
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self._ro_gt_boxes: blobs['ro_gt_boxes'],self.iter:iter}
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box, summary, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
             self._losses['rpn_loss_box'],
             self._losses['cross_entropy'],
             self._losses['loss_box'],
             self._losses['loss_box_r'],
             self._losses['uplevel_cls'],
             self._losses['attention_loss'],
             self._losses['total_loss'],
             self._show_pred_box,
             self._summary_op,
             train_op],
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box, summary, run_metadata

        # if cfg.FLAGS.use_uplevel_labels == True:
        #     if cfg.FLAGS.attention:
        #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #         run_metadata = tf.RunMetadata()
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box, summary, _ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #              self._losses['rpn_loss_box'],
        #              self._losses['cross_entropy'],
        #              self._losses['loss_box'],
        #              self._losses['loss_box_r'],
        #              self._losses['uplevel_cls'],
        #              self._losses['attention_loss'],
        #              self._losses['total_loss'],
        #              self._show_pred_box,
        #              self._summary_op,
        #              train_op],
        #             feed_dict=feed_dict,
        #             options=run_options,
        #             run_metadata=run_metadata)
        #
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, attention_loss, loss, pred_box, summary, run_metadata
        #     else:
        #         rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, loss,  smy, _ = sess.run(
        #             [self._losses["rpn_cross_entropy"],
        #             self._losses['rpn_loss_box'],
        #             self._losses['cross_entropy'],
        #             self._losses['loss_box'],
        #             self._losses['loss_box_r'],
        #              self._losses['uplevel_cls'],
        #              self._losses['total_loss'],
        #              self._summary_op,
        #              train_op],
        #              feed_dict=feed_dict)
        #         return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, uplevel_cls, loss,smy
        # else:
        #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, attention_loss, loss, summary, _ = sess.run(
        #         [self._losses["rpn_cross_entropy"],
        #          self._losses['rpn_loss_box'],
        #          self._losses['cross_entropy'],
        #          self._losses['loss_box'],
        #          self._losses['loss_box_r'],
        #          self._losses ['attention_loss'],
        #          self._losses['total_loss'],
        #          self._summary_op,
        #          train_op],
        #         feed_dict=feed_dict)
        #     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_box_r, attention_loss, loss, summary

    def train_step_no_return(self, sess, blobs, iter, train_op):
        img, h, r = flip(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['ro_gt_boxes'])

        iter = np.array([iter])
        feed_dict = {self._image: img, self._im_info: blobs['im_info'],
                     self._gt_boxes: h, self._ro_gt_boxes: r, self.iter: iter}
        sess.run([train_op], feed_dict=feed_dict)
def assert_loss_too_high(input):
    assert input<5.0
    return input