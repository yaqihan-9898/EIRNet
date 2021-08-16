import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}

network='DOSR'

######################
# General Parameters #
######################
tf.app.flags.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")

######################
# Dataset Parameters #
######################
tf.app.flags.DEFINE_string('dataset', "DOSR", "The name of dataset")
tf.app.flags.DEFINE_string('image_ext', ".png", "The name of dataset")
tf.app.flags.DEFINE_integer('uplevel_len', 5, "num of L2 upper-level classes")  # 'uplevel_len' for DOSR is 5, for HRSC2016 is 6.
FLAGS2["pixel_means"] = np.array([[[92.9714, 100.1242,  97.4815]]])

######################
# Backbone Parameters #
######################
tf.app.flags.DEFINE_string('backbone', "resnet_v1_101", "backbone name")

#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_boolean('continue_train',True, "Whether to continue_train from ckpt")
tf.app.flags.DEFINE_string('my_ckpt', "./output/pre_trained.ckpt", "Pretrained network weights")

tf.app.flags.DEFINE_float('weight_decay', 0.0001, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")
tf.app.flags.DEFINE_float('EPSILON', 1e-7, "EPSILON")

tf.app.flags.DEFINE_integer('batch_size', 256, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iters',75000, "Max iteration")
tf.app.flags.DEFINE_integer('step_size_1', 30000, "lr = lr * 0.1 after step_size_1")
tf.app.flags.DEFINE_integer('step_size_2', 60000, "lr = lr * 0.01 after step_size_2")
tf.app.flags.DEFINE_integer('display', 10, "Iteration intervals for showing the loss during training, on command line interface")
tf.app.flags.DEFINE_integer('summary_per_iter', 1000, "Iteration intervals for summary in tensorboard")
tf.app.flags.DEFINE_integer('ims_per_batch', 1, "Images to use per minibatch")

tf.app.flags.DEFINE_string('initializer', "truncated", "Network initialization parameters")
tf.app.flags.DEFINE_string('pretrained_model', "./data/pretrained_weights/pre_trained.ckpt", "Pretrained network weights")#./data/imagenet_weights/vgg16.ckpt   resnet_v1_50.ckpt  tf-densenet121.ckpt
tf.app.flags.DEFINE_string('summary_path', "./summary", "Summary path")

tf.app.flags.DEFINE_boolean('use_all_gt', True, "Whether to use all ground truth bounding boxes for training, "
                                                "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")
tf.app.flags.DEFINE_boolean('bias_decay', False, "Whether to have weight decay on bias as well")
tf.app.flags.DEFINE_boolean('double_bias', True, "Whether to double the learning rate for bias")
tf.app.flags.DEFINE_boolean('double_head',False, "Whether to double the learning rate for network head")
tf.app.flags.DEFINE_boolean('double_cls',False, "Whether to double the learning rate for cls loss")
tf.app.flags.DEFINE_integer('double_iter',30000, "If double_cls is Ture, learning rate for cls loss will double after double_iter")


#######################
# Network Parameters #
#######################
tf.app.flags.DEFINE_boolean('add_fusion',True, "Whether to ues DFF-Net")
tf.app.flags.DEFINE_boolean('attention',True, "Whether to ues add DMAM")
tf.app.flags.DEFINE_boolean('use_uplevel_labels', False, "Whether to use upper-level labels")
tf.app.flags.DEFINE_boolean('use_roi_from_att',False, "Whether to use Mask-RPN. Only be valid when attention is Ture")
tf.app.flags.DEFINE_boolean('use_2fc',False, "Whether to use 2fc layer as network head")
tf.app.flags.DEFINE_boolean('use_incep',False, "Whether to use inception layer as network head")
tf.app.flags.DEFINE_boolean('pre_ro', True, "Whether to predict rotated box ")
tf.app.flags.DEFINE_boolean('MAX_POOL', True, "Whether to max pooling")

tf.app.flags.DEFINE_integer('FIXED_BLOCKS',1, "Number of frozen backbone blocks")
tf.app.flags.DEFINE_integer('max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('test_max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('snapshot_iterations', 5000, "Iteration to take snapshot")

FLAGS2["scales"] = (600,)
FLAGS2["test_scales"] = (600,)


#######################
# Anchor Parameters #
#######################
ANCHOR_RATIOS = [0.5, 1., 2.0,3.0,1/3.0,4.0,1/4.0]
USE_CENTER_OFFSET = False
ANCHOR_SCALE_FACTORS = None

BASE_ANCHOR_SIZE_LIST = [64, 128, 256,512]
LEVLES = ['P2', 'P3','P4','P5']
ANCHOR_SCALES = [1.0]
ANCHOR_NUMS=len(ANCHOR_RATIOS)*len(ANCHOR_SCALES)
ANCHOR_STRIDE_LIST= [4, 8, 16, 32]


######################
# Testing Parameters #
######################
tf.app.flags.DEFINE_string('test_mode', "top", "Test mode for bbox proposal")

tf.app.flags.DEFINE_float('score_filter',0.001, "Only score above score_filter will be kept in output results")
tf.app.flags.DEFINE_float('RNMS_threshold',0.15, "Only score above score_filter will be kept in output results")
tf.app.flags.DEFINE_float('confidence_score',0.5, "Only score above confidence_score will plotted")

##################
# RPN Parameters #
##################
tf.app.flags.DEFINE_boolean('SHARE_HEADS', True, "Whether to SHARE_rpn_HEADS")

tf.app.flags.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
tf.app.flags.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
tf.app.flags.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")
tf.app.flags.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_float('rpn_test_nms_thresh', 0.5, "NMS threshold used on RPN proposals")

tf.app.flags.DEFINE_integer('rpn_train_pre_nms_top_n', 12000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_train_post_nms_top_n', 2000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_pre_nms_top_n', 6000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_post_nms_top_n', 600, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")
tf.app.flags.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')
tf.app.flags.DEFINE_integer('rpn_top_n', 600, "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")

tf.app.flags.DEFINE_boolean('rpn_clobber_positives', False, "If an anchor satisfied by positive and negative conditions set to negative")
IS_FILTER_OUTSIDE_BOXES = False
TRAIN_RPN_CLOOBER_POSITIVES =False
#######################
# Proposal Parameters #
#######################
tf.app.flags.DEFINE_float('proposal_fg_fraction', 0.25, "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
tf.app.flags.DEFINE_boolean('proposal_use_gt', False, "Whether to add ground truth boxes to the pool when sampling regions")

###########################
# Bounding Box Parameters #
###########################
tf.app.flags.DEFINE_float('roi_fg_threshold', 0.5, "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
tf.app.flags.DEFINE_float('roi_bg_threshold_high', 0.5, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
tf.app.flags.DEFINE_float('roi_bg_threshold_low', 0.1, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

tf.app.flags.DEFINE_boolean('bbox_normalize_targets_precomputed', True, "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
tf.app.flags.DEFINE_boolean('test_bbox_reg', True, "Test using bounding-box regressors")

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_inside_weights_ro"] = (1.0, 1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)
FLAGS2["bbox_normalize_means_ro"] = (0.0, 0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds_ro"] = (0.1, 0.1, 0.1, 0.1, 0.1)

##################
# ROI Parameters #
##################
tf.app.flags.DEFINE_integer('roi_pooling_size', 7, "Size of the pooled region after RoI pooling")

######################
# Path Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))


def get_output_dir():
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(FLAGS2["root_dir"], FLAGS2["root_dir"] , 'output','model', network))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
