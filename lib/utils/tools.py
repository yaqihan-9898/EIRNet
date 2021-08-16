# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import tensorflow as tf
import tfplot as tfp

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_heatmap(feature_maps, name):
    '''

    :param feature_maps:[B, H, W, C]
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = tf.reduce_sum(feature_maps, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    tfp.summary.plot(name, figure_attention, [heatmap])