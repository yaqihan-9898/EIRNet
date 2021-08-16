import numpy as np
import lib.config.config as cfg
import tensorflow as tf
import cv2
from random import choice

def add_gaussian_noise(image_in, noise_sigma=[5],p=[0.5]):
    """
    给图片添加高斯噪声
    image_in:输入图片
    noise_sigma：
    """
    r = np.random.randn(1)
    if r<p[0]:
        temp_image = np.float64(np.copy(image_in))
        _,h, w, _ = temp_image.shape
        noise = np.random.randn(h, w) * noise_sigma[0]
        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:,:, :, 0] = temp_image[:,:, :, 0] + noise
            noisy_image[:,:, :, 1] = temp_image[:,:, :, 1] + noise
            noisy_image[:,:, :, 2] = temp_image[:,:, :, 2] + noise
    # elif r<(p[1]+p[1]):
    #     temp_image = np.float64(np.copy(image_in))
    #     _,h, w, _ = temp_image.shape
    #     noise = np.random.randn(h, w) * noise_sigma[1]
    #     noisy_image = np.zeros(temp_image.shape, np.float64)
    #     if len(temp_image.shape) == 2:
    #         noisy_image = temp_image + noise
    #     else:
    #         noisy_image[:,:, :, 0] = temp_image[:,:, :, 0] + noise
    #         noisy_image[:,:, :, 1] = temp_image[:,:, :, 1] + noise
    #         noisy_image[:,:, :, 2] = temp_image[:,:, :, 2] + noise
    else:
        noisy_image=image_in
    return noisy_image


def short_side_resize(im, info, gtboxes_and_label_h, gtboxes_and_label_r,IMG_SHORT_SIDE_LEN = [500,600,700,800, 900,1000]):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9]
    :param target_shortside_len:
    :return:
    '''
    target_shortside_len = choice(IMG_SHORT_SIDE_LEN)
    h= info[0][0]
    w=info[0][1]
    if h<w:
        # new_h=target_shortside_len
        # new_w=target_shortside_len * w/h

        info[0][2]=info[0][2]*target_shortside_len/h
    else:
        # new_h=target_shortside_len * h/w
        # new_w=target_shortside_len
        info[0][2] = info[0][2] * target_shortside_len / w
    im = cv2.resize(im[0], None, None, fx=info[0][2], fy=info[0][2],
                    interpolation=cv2.INTER_LINEAR)
    gtboxes_and_label_r[:, 0:8] = gtboxes_and_label_r[:, 0:8] * info[0][2]
    gtboxes_and_label_r[:, 8] = gtboxes_and_label_r[:,8]
    gtboxes_and_label_h[:, 0:4] = gtboxes_and_label_h[:, 0:4] * info[0][2]
    gtboxes_and_label_h[:, 4] = gtboxes_and_label_h[:,4]
    im = np.expand_dims(im,axis=0)
    info[0][0] = im.shape[1]
    info[0][1] = im.shape[2]
    return im,info,gtboxes_and_label_h, gtboxes_and_label_r
           # tf.reshape(tf.stack([xmin,ymin,xmax,ymax,label]),[None,5]),tf.reshape(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label]),[None,9])


def box_h(xx,yy,label,new_h,new_w):
    xmin=np.min(xx)
    xmax=np.max(xx)
    ymin=np.min(yy)
    ymax=np.max(yy)
    box = np.stack([xmin, ymin, xmax, ymax,label])
    info=np.stack([new_h, new_w, 3])
    print(box.shape)
    return box,info


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len*w//h),
                               false_fn=lambda: (target_shortside_len*h//w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]








