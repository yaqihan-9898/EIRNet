import numpy as np
import cv2
def flip(img, info, hg, r):
    rk = np.random.randn(1)
    if rk < 0.5:
        temp_image = np.float64(np.copy(img))
        _, h, w, _ = temp_image.shape

        f_img=cv2.flip(temp_image[0],0).reshape([1,h,w,3])
        oldx1 = hg[:, 1].copy()
        oldx2 = hg[:, 3].copy()
        hg[:, 1] = h - oldx2 - 1
        hg[:, 3] = h - oldx1 - 1
        ro_boxes = r.copy()
        oldx1 = ro_boxes[:, 1].copy()  # xmin
        oldx2 = ro_boxes[:, 3].copy()  # xmax
        oldx3 = ro_boxes[:, 5].copy()  # xmin
        oldx4 = ro_boxes[:, 7].copy()  # xmax
        r[:, 1] = h - oldx1 - 1  # xmin= w-xax-1
        r[:, 3] = h - oldx2 - 1  # xmax =w -xmin-1
        r[:, 5] = h - oldx3 - 1  # xmin= w-xax-1
        r[:, 7] = h - oldx4 - 1  # xmax =w -xmin-1
    else:
        f_img=img

    return f_img,hg,r