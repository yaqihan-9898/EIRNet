import numpy as np
import cv2
import tensorflow as tf
import skimage.measure as measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_mask(img, boxes):
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    shape = boxes.shape
    for i in range(shape[0]):
        b = boxes[i]
        b = np.reshape(b[0:-1], [4, 2])
        # for j in range(4):
        #     if b[j,0]>60000:
        #         b[j, 0]=0
        b = b[[0,1,3,2],:]
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # for b in boxes:
    #     b = np.reshape(b[0:-1], [4, 2])
    #     rect = np.array(b, np.int32)
    #     cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    # print(mask.shape)
    return np.array(mask, np.float32)

def get_mask_plus(img, boxes):
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    shape = boxes.shape
    for i in range(shape[0]):
        b = boxes[i]
        b = np.reshape(b[0:-1], [4, 2])
        # for j in range(4):
        #     if b[j,0]>60000:
        #         b[j, 0]=0
        # b = b[[0,1,3,2],:]
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # for b in boxes:
    #     b = np.reshape(b[0:-1], [4, 2])
    #     rect = np.array(b, np.int32)
    #     cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    # print(mask.shape)
    return np.array(mask, np.float32)

def get_mask_region(img):
    img=np.squeeze(img)
    max_value = np.max(img)-1
    ret, ee = cv2.threshold(img, 1+0.45*max_value, 1, cv2.THRESH_BINARY)
    labeled_img, num = measure.label(ee, neighbors=4, background=0, return_num=True)
    i=0

    for region in measure.regionprops(labeled_img):
        # if region.area < 15:
        #     continue
            # print(regionprops(labeled_img)[max_label])
        # print(region.bbox)
        minr, minc, maxr, maxc = region.bbox
        if i==0:
            boxes=np.array([minr, minc, maxr, maxc])
        else:

            box = np.array([minr, minc, maxr, maxc])
            # print(box)
            boxes=np.vstack((boxes,box))

        # print(region)
        # boxes = boxes.append([minr, minc, maxr, maxc])
        i=i+1
        # rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                           fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)
    # print(boxes)
    boxes=boxes.astype(np.float32)
    return boxes














