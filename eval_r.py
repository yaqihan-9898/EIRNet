from lib.datasets.voc_eval_r import voc_eval
from lib.datasets import pascal_voc
from lib.config import config as cfg

import os
import pickle
import numpy as np



def get_path(image_set):
    filename = image_set+'_{:s}.txt'
    path = os.path.join(
        './output/dets_r/' + cfg.network,
        filename)
    return path


def do_python_eval(image_set, output_dir='output'):
    devkit_path = './data/VOCdevkit2007'
    year = '2007'

    annopath = './data/VOCdevkit2007/VOC2007' + '/Annotations/' + '{:s}.xml'
    imagesetfile = os.path.join(
        devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        image_set + '.txt')

    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []

    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(pascal_voc.CLASSES):
        if cls == '__background__':
            continue
        if cls in pascal_voc.CLASSES_ignore:
            continue
        filename = get_path(image_set).format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)

        aps += [ap]
        print(('AP for {} = {:.4f}'.format(cls, ap)))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print(('{:.2f}'.format(ap*100)))
    print(('{:.2f}'.format(np.mean(aps)*100)))
    print('~~~~~~~~')


if __name__ == '__main__':
    do_python_eval('test')
