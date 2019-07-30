# Mobilenetv2-Yolov3
Tensorflow implementation mobilenetv2-yolov3 inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3.git)

---
# Update
Backend:
- [x] MobilenetV2
- [x] Efficientnet
- [x] Darknet53

Callback:
- [x] mAP
- [ ] Tensorboard extern callback

Loss:
- [x] MSE
- [x] GIOU

Train:
- [x] Multi scale image size
- [x] Cosine learning rate

Tensorflow:
- [x] Tensorflow2 Ready
- [x] Use tf.data to load dataset
- [x] Use tfds to load dataset
- [x] Remove image shape input when use session
- [ ] Convert model to tensorflow lite model
- [x] Multi GPU training
- [ ] TPU support

Serving:
- [x] Tensorflow Serving warm up request
- [x] Tensorflow Serving JAVA Client
- [x] Tensorflow Serving Python Client
- [x] Tensorflow Serving Service Control Client
- [ ] Tensorflow Serving Server Build and Plugins develop 
---

# Usage
### Install:
``` bash
pip install -r requirements.txt
```
### Get help info:
``` bash
python main.py --help
```
### Train:
First: Format file name like <name>_<number>.<extension> <br>
Second: If you are using txt dataset, please format records like <image_path> [<xmin> <ymin> <xmax> <ymax> <class>] <br>(for convenience, you can modify voc_text.py to parse your data to specific data format),else you should use modify voc_annotation.py,then run 
``` bash
python voc_annotation.py
``` 
to parse your data to tfrecords.
Third: Run
``` bash
python main.py --mode=TRAIN --train_dataset_glob=<your dataset glob>
```

### Predict:
``` bash
python main.py --mode=IMAGE --model=<your_model_path>
```
### Export serving model:
``` bash
python main.py --mode=SERVING --model=<your_model_path>
```
### Use custom config file:
``` bash
python main.py --config=mobilenetv2.yaml
```

---

# Resources
* Download pascal tfrecords from [here](https://drive.google.com/drive/folders/172sH75LPeUd2yyzAnrce0LLe2UR_kFqF).
* Download pre-trained mobilenetv2-yolov3 model(VOC2007+VOC2012) [here](https://drive.google.com/open?id=1B0vVQsuWY-zfuyol38-R5XJs1mntIwqZ)

---

# Performance
Network: Mobilenetv2+Yolov3 <br>
Input size: 416*416 <br>
Dataset: VOC2007 <br>
mAP: <br>
```
aeroplane ap:  0.585270123970341
bicycle ap:  0.7311717479746895
bird ap:  0.6228634475289679
boat ap:  0.44729361226611786
bottle ap:  0.3524265151288485
bus ap:  0.7260233058709467
car ap:  0.7572503412774444
cat ap:  0.8443930169586521
chair ap:  0.3530240979604032
cow ap:  0.5680746465428056
diningtable ap:  0.6046673143934721
dog ap:  0.8096497542858805
horse ap:  0.785358647511358
motorbike ap:  0.7299038925396009
person ap:  0.6926967393665762
pottedplant ap:  0.2960290730045794
sheep ap:  0.5569735405574012
sofa ap:  0.6053534702293342
train ap:  0.7304618425853895
tvmonitor ap:  0.5983913977616169
mAP:  0.6198638263857212
```
GPU inference time (Python+MX150): 78ms <br>
CPU inference time (Python+i7-8550U): 112ms <br>
Model size: 37M <br>

---

# Reference
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