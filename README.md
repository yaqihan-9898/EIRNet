# EIRNet

Our paper "Fine-Grained Recognition for Oriented Ship Against Complex Scenes in Optical Remote in Remote Sensing Images" is under submission. We decide to publish our code in advance.
Note: This is an early version of the code. We will update it in the future.

Introduction
--
Efficient Information Reuse Network (EIRNet) is a two-stage end-to-end network designed for fine-grained ship recognition in remote sensing images. We will update the detaied structure of EIRNet after the acception of our manuscript :)

We tested EIRNet on two datasets: HRSC2016 and [DOSR][1]. HRSC2016 is a widely used dataset for fine-grained ship recognition. DOSR is a new fine-grained ship recognition dataset collected by ourselves with more complex scenes. Extensive experiments demonstrate our EIRNet achieves state-of-the-art performance on DOSR and HRSC2016.

Experiment Environment
--
Ubuntu / Win10

Python 3.5

Tensorflow 1.10.0

Cuda 9.0

GeForce GTX 1080Ti


How to use
--

### 1. Download the code and prepare for training.

```
$ git clone https://github.com/ultralytics/yolov5  # clone repo
$ cd EIRNet
$ pip install -r requirements.txt  # install
```

### 2. Compile

For linux:
```
$ cd ./data/coco/PythonAPI
$ python setup.py build_ext --inplace
$ python setup.py build_ext install
$ cd ./lib/utils　　
$ python setup.py build_ext --inplace
```
For windows:
```
$ cd /d .\data\coco\PythonAPI
$ python setup.py build_ext --inplace
$ python setup.py build_ext install
$ cd /d .\lib\utils　　
$ python setup.py build_ext --inplace
```
### 3. Prepare data and pretrained model

Put all dataset images under ./data/VOCdevkit2007/VOC2007/JPEGImages
Put all dataset annotations (VOC format) under ./data/VOCdevkit2007/VOC2007/Annotations
Put train.txt, val.txt, test.txt, and trainval.txt under ./data/VOCdevkit2007/VOC2007/ImageSets/Main

Download pretrained weight from [our pretrained model][2] (only ResNet101) or [office model][3].

### 4. Train
Modify ./lib/config/config.py:
'network' in Line10

If you want to train your own dataset, please modify:
./lib/config/config.py: 'dataset' in Line20, 'image_ext' in Line21, 'uplevel_len' in Line22
./lib/datasets/pascal_voc.py: 'CLASSES' in Line24
./lib/layer_utils/uplevel_utils.py: 'uplevelmap' in Line4
./lib/layer_utils/cls_weight.py: 'cls_weight_L2' in Line4

The default cache files in ./data/cache and ./data/VOCdevkit2007/annotations_cache is obtained from DOSR. 
If you want to train or test other datasets, please delete ./data/cache/HRSC2016/voc_2007_trainval_gt_roidb.pkl and ./data/VOCdevkit2007/annotations_cache/annots.pkl


Go to the root path and start to train:
```
$ python train.py
```

Note: At the beginning of the train, there may be some assert errors. The reason is there is an assert 'fg_num>0' in function 'iou_rotate_calculate3' of ./lib/utils/iou_rotate.py. At the beginning of the train, the network gets no foreground，thus, network cannot comput IoU between the predicted foreground box and its corresponding groundtruth bounding box. Such assert errors will not affect training process, you can ignore it. You can use our pretrained model to reduce such assert errors.
### 5. Demo
```
$ python demo.py
```

### 6. Eval
```
$ python eval_r.py
```


[1]:https://github.com/yaqihan-9898/DOSR
[2]: https://pan.baidu.com/s/1I7N0I_y2en2_PyYhMzuScg
[3]: https://github.com/tensorflow/models/tree/master/research/slim
