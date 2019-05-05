# mobilenetv2-yolov3
Tensorflow implementation mobilenetv2-yolov3 inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3.git)

# Update
Backend:
- [x] MobilenetV2
- [x] Densenet
- [x] Darknet53
- [ ] Inception-ResV2

Callback:
- [x] mAP
- [ ] Tensorboard extern callback

Loss:
- [x] MSE
- [x] GIOU
- [x] Focal Loss

Train:
- [x] Multi scale image size
- [x] Cosine learning rate

Tensorflow:
- [x] Tensorflow2 Ready
- [x] Use tf.data to load dataset
- [x] Remove image shape input when use session
- [ ] Convert model to tensorflow lite model
- [ ] Multi GPU training support
- [x] Tensorflow Serving JAVA Client
- [ ] Tensorflow Serving Server Build and Plugins develop 

# Requirement
Tensorflow-1.13+

Numpy-1.16.2+

Python-3.6.7+

# Usage
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
  --export           Export binary pb model for tensorflow,which you can put it in tensorflow serving directly
```
---

# Train
* Download pascal tfrecords from [here](https://drive.google.com/drive/folders/172sH75LPeUd2yyzAnrce0LLe2UR_kFqF).
* Change [train.py](https://github.com/fsx950223/mobilenetv2-yolov3/blob/master/train.py)

```
opt = <your session config>
backbone = <your yolov3 backbone>
log_dir = <path/to/your/tensorboard/log>
batch_size = <you batch size>
train_dataset_path = <path/to/your/train/folder>
val_dataset_path = <path/to/your/val/folder>
train_dataset_glob = <train glob>
val_dataset_glob = <val glob>
```
---

# Performance
3 times faster than darknet53-yolov3 with alpha=1.4 and higher accuracy

# Pascal Dataset
I have packaged a pascal tfrecords for you.See [here](https://drive.google.com/drive/folders/172sH75LPeUd2yyzAnrce0LLe2UR_kFqF)

# Reference:<br>
paper: <br>
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)<br>
- [Foca Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)<br>
- [Group Normalization](https://arxiv.org/abs/1803.08494)<br>
- [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)<br>
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)<br>
- [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)<br>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)<br>
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)<br>
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)<br>