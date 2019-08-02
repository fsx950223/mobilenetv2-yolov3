# Mobilenetv2-Yolov3
Tensorflow implementation mobilenetv2-yolov3 and efficientnet-yolov3 inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3.git)

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
1. Format file name like [name]_[number].[extension] <br>
Example: <br>
```
voc_train_3998.txt
```
<br>
2. If you are using txt dataset, please format records like [image_path] [,[xmin ymin xmax ymax class]] <br>(for convenience, you can modify voc_text.py to parse your data to specific data format), else you should modify voc_annotation.py, then run <br>

``` bash
python voc_annotation.py
``` 
to parse your data to tfrecords. <br>
Example: <br>

```
/image/path 179 66 272 290 14 172 38 317 349 14 276 2 426 252 14 1 32 498 365 13
```
<br>
3. Run: <br>

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
# Set up tensorflow.js model (Live Demo: https://fsx950223.github.io/mobilenetv2-yolov3/tfjs/)
1. Create a web server on project folder <br>
2. Open browser and enter [your_url:your_port]/tfjs <br>

---
# Resources
* Download pascal tfrecords from [here](https://drive.google.com/drive/folders/172sH75LPeUd2yyzAnrce0LLe2UR_kFqF).
* Download pre-trained mobilenetv2-yolov3 model(VOC2007) [here](https://drive.google.com/open?id=1B0vVQsuWY-zfuyol38-R5XJs1mntIwqZ)
* Download pre-trained efficientnet-yolov3 model(VOC2007) [here](https://drive.google.com/open?id=10A2BqNrQp5_hIcBzGXu6Xiv4mCQzga2q)
* Download pre-trained efficientnet-yolov3 model(VOC2007+2012) [here](https://drive.google.com/open?id=1dYfi1z5EeNsXMLACwoeR4jGj7RWyCcZp)

---

# Performance
Network: Mobilenetv2+Yolov3 <br>
Input size: 416*416 <br>
Train Dataset: VOC2007 <br>
Test Dataset: VOC2007 <br>
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
GPU inference time (GTX1080Ti): 19ms <br>
CPU inference time (i7-8550U): 112ms <br>
Model size: 37M <br>
<br>
Network: Efficientnet+Yolov3 <br>
Input size: 380*380 <br>
Train Dataset: VOC2007 <br>
Test Dataset: VOC2007 <br>
mAP: <br>

```
aeroplane ap:  0.6492260838166934
bicycle ap:  0.8010712280076165
bird ap:  0.7013865117634108
boat ap:  0.5557173155813903
bottle ap:  0.4353563564340365
bus ap:  0.753804699972881
car ap:  0.7878183961387922
cat ap:  0.8632726491920759
chair ap:  0.4090719340574334
cow ap:  0.6657089830054761
diningtable ap:  0.6513494390619038
dog ap:  0.8466486584164448
horse ap:  0.8328765157511936
motorbike ap:  0.7607912651726462
person ap:  0.7089970516297166
pottedplant ap:  0.32875322571854027
sheep ap:  0.6372370950276296
sofa ap:  0.675301446703759
train ap:  0.7734685594308568
tvmonitor ap:  0.6505409659737674
mAP:  0.6744199190428132
```
GPU inference time (GTX1080Ti): 23ms <br>
CPU inference time (i7-8550U): 168ms <br>
Model size: 77M <br>
<br>
Network: Efficientnet+Yolov3 <br>
Input size: 380*380 <br>
Train Dataset: VOC2007+VOC2012 <br>
Test Dataset: VOC2007 <br>
mAP: <br>

```
aeroplane ap:  0.8186380791530123
bicycle ap:  0.778370501901752
bird ap:  0.8040658409051149
boat ap:  0.6606796907615438
bottle ap:  0.5338128542328597
bus ap:  0.8516086793836817
car ap:  0.8247881435224634
cat ap:  0.9271784386863242
chair ap:  0.5344565229671414
cow ap:  0.7724057669182698
diningtable ap:  0.701598520527006
dog ap:  0.9052246177009002
horse ap:  0.8477206181813397
motorbike ap:  0.8275932123398402
person ap:  0.7605203479510053
pottedplant ap:  0.45979410517062475
sheep ap:  0.8301611044152797
sofa ap:  0.7393617389123919
train ap:  0.8817430073959469
tvmonitor ap:  0.6981047903116634
mAP:  0.757891329066908
```
GPU inference time (GTX1080Ti): 23ms <br>
CPU inference time (i7-8550U): 168ms <br>
Model size: 77M <br>

---

# Reference
paper: <br>
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)<br>
- [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)<br>
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)<br>
- [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)<br>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)<br>
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)<br>
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)<br>