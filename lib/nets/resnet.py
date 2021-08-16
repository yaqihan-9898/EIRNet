# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import lib.config.config as cfg
from lib.nets.network import Network
from lib.utils.tools import add_heatmap
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from lib.layer_utils import anchor_utils
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.get_mask import get_mask_region
from lib.layer_utils.proposal_layer import add_demension
from lib.layer_utils import rfcn_plus_plus_opr

def resnet_arg_scope(
        is_training=True, weight_decay=cfg.FLAGS.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class resnet(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)
        self._resnet_scope = cfg.FLAGS.backbone

    def build_network(self, sess, is_training=True):

        with tf.variable_scope(self._resnet_scope, self._resnet_scope):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head
        if cfg.FLAGS.attention == True:
            net, net_attention, net_attention_plus = self.build_head(is_training, initializer)

            self._predictions["net_attention"] =net_attention
            self._predictions["net_attention_plus"] = net_attention_plus
        else:
            net = self.build_head(is_training, initializer)

            # Build rpn
        with tf.variable_scope(self._resnet_scope, self._resnet_scope):

            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score= self.build_rpn(net, is_training, initializer)

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

        # Build predictions
        net = net[1]  # choose A3 layer to map features
        if cfg.FLAGS.pre_ro == True:
            if cfg.FLAGS.use_uplevel_labels == True:
                cls_score, cls_prob, bbox_pred, cls_score_uplevel, cls_prob_uplevel, ro_bbox_pred = self.build_predictions(
                    net,
                    rois,
                    is_training,
                    initializer,
                    initializer_bbox)
                self._predictions["uplevel_cls_score"] = cls_score_uplevel
                self._predictions["uplevel_cls_prob"] = cls_prob_uplevel
                self._predictions["ro_bbox_pred"] = ro_bbox_pred
                # self._predictions["uplevel_plus_cls_score"]=cls_score_uplevel_plus
                # self._predictions["uplevel_plus_cls_prob"] = cls_prob_uplevel_plus
            else:
                cls_score, cls_prob, bbox_pred, ro_bbox_pred = self.build_predictions(net, rois, is_training,
                                                                                      initializer,
                                                                                      initializer_bbox)
            self._predictions["ro_bbox_pred"] = ro_bbox_pred
        else:
            if cfg.FLAGS.use_uplevel_labels == True:
                cls_score, cls_prob, bbox_pred, cls_score_uplevel, cls_prob_uplevel = self.build_predictions(
                    net,
                    rois,
                    is_training,
                    initializer,
                    initializer_bbox)
                self._predictions["uplevel_cls_score"] = cls_score_uplevel
                self._predictions["uplevel_cls_prob"] = cls_prob_uplevel
            else:
                cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer,
                                                                        initializer_bbox)

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois

        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred


    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._resnet_scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def get_variables_to_restore_head(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if v.name.startswith('conv_new_1') and v.name.split(':')[0] in var_keep_dic:
                print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix Resnet V1 layers..')
        with tf.variable_scope('Fix_Resnet_V1') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)
                if cfg.FLAGS.continue_train:
                    sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                                       conv1_rgb))
                else:
                    sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_attention(self,inputs, is_training, initializer):
        attention_conv3x3_1 = slim.conv2d(inputs, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_1')
        attention_conv3x3_2 = slim.conv2d(attention_conv3x3_1, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_2')
        attention_conv3x3_3 = slim.conv2d(attention_conv3x3_2, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_3')
        attention_conv3x3_4 = slim.conv2d(attention_conv3x3_3, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_4')
        attention_conv3x3_5 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=None,
                                          scope='attention_conv/3x3_5')
        attention_conv3x3_6 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=None,
                                          scope='attention_conv/3x3_6')
        return attention_conv3x3_5,attention_conv3x3_6

    def fusion_two_layer(self,C_i, P_j, scope,channel):
        '''
        i = j+1
        :param C_i: shape is [1, h, w, c]
        :param P_j: shape is [1, h/2, w/2, 256]
        :return:
        P_i
        '''
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]
            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_' + level_name)


            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=channel,
                                       kernel_size=[1, 1], stride=1,
                                       scope='reduce_dim_' + level_name)
            if level_name =='P3':
                reduce_dim_c = self.cbam_block(reduce_dim_c, 'cbam_C3')
            add_f = 0.5 * upsample_p + 0.5 * reduce_dim_c

            return add_f





    def build_head(self, is_training,initializer):

        scope_name = self._resnet_scope
        if scope_name == 'resnet_v1_50':
            block2_num_units = 4
            block3_num_units = 6
        elif scope_name == 'resnet_v1_101':
            block2_num_units = 4
            block3_num_units = 23
        elif scope_name == 'resnet_v1_152':
            block2_num_units = 8
            block3_num_units = 36
        else:
            raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101 or resnet_v1_152. '
                                      'Check your network name.')

        blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=128, num_units=block2_num_units, stride=2),
                  resnet_v1_block('block3', base_depth=256, num_units=block3_num_units, stride=1)]
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            with tf.variable_scope(scope_name, scope_name):
                # Do the first few layers manually, because 'SAME' padding can behave inconsistently
                # for images of different sizes: sometimes 0, sometimes 1
                net = resnet_utils.conv2d_same(
                    self._image, 64, 7, stride=2, scope='conv1')
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                net = slim.max_pool2d(
                    net, [3, 3], stride=2, padding='VALID', scope='pool1')

        not_freezed = [False] * cfg.FLAGS.FIXED_BLOCKS + (4 - cfg.FLAGS.FIXED_BLOCKS) * [True]
        # Fixed_Blocks can be 1~3
        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
            C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                    blocks[0:1],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=scope_name)
            # add_heatmap(tf.expand_dims(tf.reduce_mean(C2, axis=-1), axis=-1),
            #            'C2')

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
            C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                    blocks[1:2],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=scope_name)
            # add_heatmap(tf.expand_dims(tf.reduce_mean(C3, axis=-1), axis=-1),
            #            'C3')

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
            C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                        blocks[2:3],
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=scope_name)

            # add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1),
            #            'C4')
            if cfg.FLAGS.add_fusion:
                # P5 = slim.max_pool2d(
                #     C4, [3, 3], stride=2, padding='VALID', scope='pool1')
                # P5 = rfcn_plus_plus_opr.global_context_module(
                #     P5, prefix='conv_new_1',
                #     ks=3, chl_mid=256, chl_out=1024, is_training=is_training)

                _C3 = slim.conv2d(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                                  self._channels, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  activation_fn=tf.nn.relu,
                                  scope='C3_conv3x3')

                # cbam_C3 = self.cbam_block(_C3, 'cbam_C3', only_channel=True)
                _C2 = slim.conv2d(end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                                  self._channels, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  activation_fn=tf.nn.relu,
                                  scope='C2_conv1x1')
                # sq_C2 = self.squeeze_excitation_layer(_C2, 1024, 16, 'SE_C2', is_training, initializer)
                cbam_C2 = self.cbam_block(_C2, 'cbam_C2', only_channel=True)
                C4_shape=tf.shape(C4)
                # C4= self.cbam_block(C4, 'cbam_C4', only_channel=True)
                # cbam_C3=_C3
                # cbam_C2 = _C2
                P4=C4+self.cbam_block(tf.image.resize_bilinear(_C3, (C4_shape[1], C4_shape[2]))+ tf.image.resize_bilinear(cbam_C2, (C4_shape[1], C4_shape[2])), 'cbam_C3', only_channel=True)
                # P4=C4+self.squeeze_excitation_layer(tf.image.resize_bilinear(_C3, (C4_shape[1], C4_shape[2]))+ tf.image.resize_bilinear(sq_C2, (C4_shape[1], C4_shape[2])), 1024, 16, 'SE_C3', is_training, initializer)

                # P4= self.cbam_block(P4, 'cbam_P4', only_channel=True)
                # P4 = C4 + tf.image.resize_bilinear(_C3, (C4_shape[1], C4_shape[2])) + tf.image.resize_bilinear(
                #     _C2, (C4_shape[1], C4_shape[2]))

                C3_shape = tf.shape(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)])
                _P4 = tf.image.resize_bilinear(P4, (C3_shape[1], C3_shape[2]))
                _C2_1 = tf.image.resize_bilinear(cbam_C2, (C3_shape[1], C3_shape[2]))
                # _C2_1 = tf.image.resize_bilinear(sq_C2, (C3_shape[1], C3_shape[2]))
                # _C3 = self.squeeze_excitation_layer(_C3, 1024, 16, 'SE_C3', is_training, initializer_bbox)
                # _C3 = self.cbam_block(_C3,'cbam_C3',only_channel=True)
                #  add_heatmap(tf.expand_dims(tf.reduce_mean(_C3, axis=-1), axis=-1),
                #              'C3_after_cbam')
                #  _C3 = self.cbam_block(_C3, 'cbam_C3', only_channel=True)
                P3 = _P4 + _C3+_C2_1
                # P3 = self.cbam_block(P3, 'cbam_P3', only_channel=True)
                # P3 = _P4 + _C3+ self.cbam_block(_C2_1, 'cbam_C2', only_channel=True)
                C2_shape = tf.shape(end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)])
                _P3 = tf.image.resize_bilinear(P3, (C2_shape[1], C2_shape[2]))
                _C4 = tf.image.resize_bilinear(P4, (C2_shape[1], C2_shape[2]))

                P2=_C2+_P3+_C4
                # P2 = self.cbam_block(P2, 'cbam_P2', only_channel=True)
                # add_heatmap(tf.expand_dims(tf.reduce_mean(P2, axis=-1), axis=-1),
                #              'P2')
                # add_heatmap(tf.expand_dims(tf.reduce_mean(P3, axis=-1), axis=-1),
                #             'P3')
                # add_heatmap(tf.expand_dims(tf.reduce_mean(P4, axis=-1), axis=-1),
                #             'P4')
                # P_avr = tf.image.resize_bilinear(P4, (C3_shape[1], C3_shape[2])) + P3 + tf.image.resize_bilinear(P2, (
                # C3_shape[1], C3_shape[2]))
                # P_avr = P_avr / 3.0
                # Make Gaussian Kernel with desired specs.
                # P_avr = slim.conv2d(P_avr,
                #                   1024, [3, 3],
                #                   trainable=is_training,
                #                   weights_initializer=initializer,
                #                   activation_fn=tf.nn.relu,
                #                   scope='P_avr')  # 1024
                # # Convolution kernel
                # kernel = make_gaussian_2d_kernel(5)
                # kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
                # P_avr=tf.nn.conv2d(P_avr, kernel, strides=[1, 1, 1, 1], padding="SAME")

                # P_avr = tf.nn.separable_conv2d(
                #     P_avr, kernel, tf.eye(3, batch_shape=[1, 1]),
                #     strides=[1, 1, 1, 1], padding='SAME')


                # P4=P4+tf.image.resize_bilinear(P_avr, (C4_shape[1], C4_shape[2]))
                # P3=P3+P_avr
                # P2 = P2 + tf.image.resize_bilinear(P_avr, (C2_shape[1], C2_shape[2]))

            if cfg.FLAGS.attention == True:
                with tf.variable_scope('build_attention',
                                       regularizer=slim.l2_regularizer(cfg.FLAGS.weight_decay)):

                    add_heatmap(tf.expand_dims(tf.reduce_mean(P3, axis=-1), axis=-1), 'add_attention_before')
                    mask1,mask2 = self.build_attention(P3, is_training,initializer)
                    # mask1 = self.build_inception_attention(C4_stop, is_training,initializer)
                    mask1_attention = tf.nn.softmax(mask1)
                    mask2_attention = tf.nn.softmax(mask2)
                    # mask1_attention = tf.Print(mask1_attention, [tf.shape(mask1_attention)], summarize=10,
                    #                               message='mask1_attention')
                    # mask2_attention = tf.Print( mask2_attention, [tf.shape( mask2_attention)],
                    #                                    summarize=10, message=' mask2_attention')
                    mask1_attention = mask1_attention[:, :, :, 1]
                    mask2_attention = mask2_attention[:, :, :, 1]
                    mask1_attention = tf.expand_dims(mask1_attention, axis=-1)
                    add_heatmap(mask1_attention, 'mask1_attention')
                    mask2_attention = tf.expand_dims( mask2_attention, axis=-1)
                    add_heatmap(mask2_attention, 'mask2_attentionn')

                    C3_attention = 1 + 0.5*mask1_attention + 0.5*mask2_attention
                    self.attention = C3_attention
                    C2_attention = tf.image.resize_bilinear(C3_attention, (C2_shape[1], C2_shape[2]))

                    C4_shape = tf.shape(C4)
                    C4_attention = tf.image.resize_bilinear(C3_attention, (C4_shape[1], C4_shape[2]))
                    add_heatmap(C3_attention, 'final_attention')
                    
                    P2 = tf.multiply(C2_attention, P2)
                    P3 = tf.multiply(C3_attention, P3)
                    P4 = tf.multiply(C4_attention, P4)
                    add_heatmap(tf.expand_dims(tf.reduce_mean(P3, axis=-1), axis=-1), 'add_attention_after')
                    P5 = slim.max_pool2d(
                        C4, [3, 3], stride=2, padding='VALID', scope='pool1')
                    C5_attention = slim.max_pool2d(
                        C4_attention, [3, 3], stride=2, padding='VALID', scope='pool1')

                    self.ATTENTION_DIC = {'P2': C2_attention, 'P3': C3_attention, 'P4': C4_attention,
                                          'P5': C5_attention}

                    self.mask_boxes = \
                        tf.py_func(
                            get_mask_region,
                            [self.attention[0]],
                            [tf.float32])
                    self.mask_boxes=tf.reshape(self.mask_boxes,[-1,4])
            else:
                P5 = slim.max_pool2d(
                    C4, [3, 3], stride=2, padding='VALID', scope='pool1')

        if cfg.FLAGS.attention == True:
            return [P2,P3,P4,P5], mask1,mask2
        else:
            return [P2,P3,P4,P5]



    def build_rpn(self, net, is_training, initializer):
        all_anchors = []
        for i in range(len(cfg.LEVLES)):
            level_name, p = cfg.LEVLES[i], net[i]

            p_h, p_w = tf.shape(p)[1], tf.shape(p)[2]
            featuremap_height = tf.cast(p_h, tf.float32)
            featuremap_width = tf.cast(p_w, tf.float32)
            anchors = anchor_utils.make_anchors(base_anchor_size=cfg.BASE_ANCHOR_SIZE_LIST[i],
                                                anchor_scales=cfg.ANCHOR_SCALES,
                                                anchor_ratios=cfg.ANCHOR_RATIOS,
                                                featuremap_height=featuremap_height,
                                                featuremap_width=featuremap_width,
                                                stride=cfg.ANCHOR_STRIDE_LIST[i],
                                                name="make_anchors_for%s" % level_name)
            all_anchors.append(anchors)
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors_of_FPN')
        self._anchors = all_anchors

        fpn_cls_score = []
        fpn_box_pred = []

        for level_name, p in zip(cfg.LEVLES, net):
            if cfg.FLAGS.SHARE_HEADS:
                reuse_flag = None if level_name == cfg.LEVLES[0] else True
                scope_list = ['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']
            else:
                reuse_flag = None
                scope_list = ['rpn_conv/3x3_%s' % level_name, 'rpn_cls_score_%s' % level_name,
                              'rpn_bbox_pred_%s' % level_name]
            rpn = slim.conv2d(p, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                              scope=scope_list[0], reuse=reuse_flag)
            rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], stride=1, trainable=is_training,
                                        weights_initializer=initializer, padding='VALID', activation_fn=None,
                                        scope=scope_list[1], reuse=reuse_flag)  # stride =2 ?

            rpn_box_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], stride=1, padding='VALID',
                                       trainable=is_training, weights_initializer=initializer, activation_fn=None,
                                       scope=scope_list[2], reuse=reuse_flag)

            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

            fpn_cls_score.append(rpn_cls_score)
            fpn_box_pred.append(rpn_box_pred)

        fpn_cls_score = tf.concat(fpn_cls_score, axis=0, name='fpn_cls_score')
        fpn_box_pred = tf.concat(fpn_box_pred, axis=0, name='fpn_box_pred')
        fpn_cls_prob = slim.softmax(fpn_cls_score, scope='fpn_cls_prob')

        return fpn_cls_prob, fpn_box_pred, fpn_cls_score





    def build_proposals(self, is_training, fpn_cls_prob, fpn_box_pred, rpn_cls_score):
        with tf.variable_scope('postprocess_FPN'):
            rois, roi_scores = self._proposal_layer(fpn_cls_prob, fpn_box_pred, "rois")

            if not is_training:
                if cfg.FLAGS.use_roi_from_att:
                    cols = tf.unstack(self.mask_boxes, axis=1)

                    boxes = tf.stack([(cols[1] + cols[3]) / 2,
                                      (cols[0] + cols[2]) / 2,
                                      cols[3] + cols[1],
                                      cols[2] + cols[0]], axis=1)  # (?, 4)
                    boxes = tf.py_func(add_demension, [boxes], [tf.float32])
                    boxes = tf.reshape(boxes, [-1, 5])
                    rois = tf.concat([rois, boxes], axis=0)
        if is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                fpn_labels, fpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [self._gt_boxes, self._im_info, self._anchors, self._num_classes],
                        [tf.float32, tf.float32])
                fpn_bbox_targets = tf.reshape(fpn_bbox_targets, [-1, 4])  # loss用
                fpn_labels = tf.to_int32(fpn_labels, name="to_int32")
                fpn_labels = tf.reshape(fpn_labels, [-1])

                self._anchor_targets['rpn_labels'] = fpn_labels
                self._anchor_targets['rpn_bbox_targets'] = fpn_bbox_targets

            with tf.control_dependencies([fpn_labels]):
                if cfg.FLAGS.use_roi_from_att:
                    cols = tf.unstack(self.mask_boxes, axis=1)

                    boxes = tf.stack([(cols[1] + cols[3]) / 2,
                                      (cols[0] + cols[2]) / 2,
                                      cols[3] + cols[1],
                                      cols[2] + cols[0]], axis=1)  # (?, 4)
                    boxes = tf.py_func(add_demension, [boxes], [tf.float32])
                    boxes = tf.reshape(boxes, [-1, 5])
                    # rois = tf.concat([rois, boxes], axis=0)
                    rois_from_att = tf.reshape(boxes, [-1, 5])
                else:
                    rois_from_att = rois[0]
                    rois_from_att = tf.reshape(rois_from_att, [-1, 5])
                rois, _ = self._proposal_target_layer(rois, roi_scores, rois_from_att, "rpn_rois")
        return rois


    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        if cfg.FLAGS.use_2fc:
            with tf.variable_scope('fc', 'fc'):
                pool5 = self._crop_pool_layer(net, rois, "pool5")
                inputs = slim.flatten(inputs=pool5, scope='flatten_inputs')
                fc6 = slim.fully_connected(inputs, num_outputs=1024, scope='fc6')
                if is_training:
                    fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout1')
                fc7 = slim.fully_connected(fc6, num_outputs=1024, scope='fc7')
                if is_training:
                    net_to_predict = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
        elif cfg.FLAGS.use_incep:
            with tf.variable_scope('inception', 'inception'):
                pool5 = self._crop_pool_layer(net, rois, "pool5")
                featrue_map=self.build_inception(pool5,is_training,initializer,'incp1')
                featrue_map = self.build_inception(featrue_map,is_training,initializer,'incp2')
                featrue_map = self.build_inception(featrue_map,is_training,initializer,'incp3')
                featrue_map = self.build_inception(featrue_map,is_training,initializer,'incp4')
                net_to_predict = tf.reduce_mean(featrue_map, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        else:
            scope_name = self._resnet_scope
            pool5 = self._crop_pool_layer(net, rois, "pool5")
            block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                C5, _ = resnet_v1.resnet_v1(pool5,
                                            block4,
                                            global_pool=False,
                                            include_root_block=False,
                                            scope=scope_name)
                net_to_predict = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')


        # Scores and predictions
        with tf.variable_scope('faster_rcnn', 'faster_rcnn'):
            cls_score = slim.fully_connected(net_to_predict, self._num_classes, weights_initializer=initializer,
                                             trainable=is_training, activation_fn=None, scope='cls_score')
            cls_prob = self._softmax_layer(cls_score, "cls_prob")
            bbox_prediction = slim.fully_connected(net_to_predict, self._num_classes * 4, weights_initializer=initializer_bbox,
                                                   trainable=is_training, activation_fn=None, scope='bbox_pred')
            if cfg.FLAGS.pre_ro == True:
                ro_bbox_prediction = slim.fully_connected(net_to_predict, self._num_classes * 5,
                                                          weights_initializer=initializer_bbox,
                                                          trainable=is_training, activation_fn=None,
                                                          scope='ro_bbox_pred')
            if cfg.FLAGS.use_uplevel_labels == True:
                cls_score_uplevel = slim.fully_connected(net_to_predict, cfg.FLAGS.uplevel_len, weights_initializer=initializer,
                                                         trainable=is_training,
                                                         activation_fn=None, scope='uplevel_cls_score')
                cls_prob_uplevel = self._softmax_layer(cls_score_uplevel, "uplevel_cls_prob")
                # cls_score_uplevel_plus = slim.fully_connected(fc7, 2, weights_initializer=initializer,
                #                                          trainable=is_training,
                #                                          activation_fn=None, scope='uplevel_cls_score_plus')
                # cls_prob_uplevel_plus = self._softmax_layer(cls_score_uplevel_plus, "uplevel_cls_prob_plus")

            if cfg.FLAGS.pre_ro == True:
                if cfg.FLAGS.use_uplevel_labels == True:
                    return cls_score, cls_prob, bbox_prediction, cls_score_uplevel, cls_prob_uplevel,ro_bbox_prediction
                else:
                    return cls_score, cls_prob, bbox_prediction, ro_bbox_prediction
            else:
                if cfg.FLAGS.use_uplevel_labels == True:
                    return cls_score, cls_prob, bbox_prediction, cls_score_uplevel, cls_prob_uplevel
                else:
                    return cls_score, cls_prob, bbox_prediction


    def build_inception(self, inputs, is_training,initializer,scope):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 256, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 224, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 256, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='avgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0b_1x1')
            inception_out = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            return inception_out

    def build_inception_attention(self,inputs, is_training,initializer):
        """Builds Inception-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        inception_out = self.build_inception(inputs, is_training,initializer)

        inception_attention_out = slim.conv2d(inception_out, 2, [3, 3],
                                              trainable=is_training,
                                              weights_initializer=initializer,
                                              activation_fn=None,
                                              scope='inception_attention_out')
        return inception_attention_out

    def squeeze_excitation_layer(self,input_x, out_dim, ratio, layer_name, is_training,initializer_bbox):
        with tf.name_scope(layer_name):
            # Global_Average_Pooling
            squeeze = tf.reduce_mean(input_x, [1, 2])

            excitation = slim.fully_connected(inputs=squeeze,
                                              num_outputs=out_dim // ratio,
                                              weights_initializer=initializer_bbox,
                                              activation_fn=tf.nn.relu,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected1')

            excitation = slim.fully_connected(inputs=excitation,
                                              num_outputs=out_dim,
                                              weights_initializer=initializer_bbox,
                                              activation_fn=tf.nn.sigmoid,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected2')

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            # scale = input_x * excitation

            return excitation

    def  cbam_block(self,input_feature, name, only_channel=False,ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """

        with tf.variable_scope(name):

            attention_feature = self.channel_attention(input_feature, 'ch_at', ratio,mode=1)
            if not only_channel:
                attention_feature = self.spatial_attention(attention_feature, 'sp_at')
            print("CBAM Hello")
        return attention_feature


    def channel_attention(self,input_feature, name, ratio=8,mode=1):

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name):
            channel = input_feature.get_shape()[-1]
            avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

            assert avg_pool.get_shape()[1:] == (1, 1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_0',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_1',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)

            max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       name='mlp_0',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel,
                                       name='mlp_1',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)

            scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
        if mode == 1:
            return input_feature * scale
        else:
            return  scale

    def spatial_attention(self,input_feature, name):
        kernel_size = 7
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            concat = tf.concat([avg_pool, max_pool], 3)
            assert concat.get_shape()[-1] == 2

            concat = tf.layers.conv2d(concat,
                                      filters=1,
                                      kernel_size=[kernel_size, kernel_size],
                                      strides=[1, 1],
                                      padding="same",
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=False,
                                      name='conv')
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, 'sigmoid')

        return input_feature * concat


def gauss(x, y, sigma=3.):
    Z = 2*np.pi*sigma**2
    return 1/Z*np.exp(-(x**2+y**2)/2/sigma**2)
