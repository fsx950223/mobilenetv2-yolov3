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
- [x] Adversarial loss

Train:
- [x] Cosine learning rate
- [x] Auto augment

Tensorflow:
- [x] Tensorflow2 Ready
- [x] tf.data pipeline
- [ ] Convert model to tensorflow lite model
- [x] Multi GPU training
- [ ] TPU support
- [x] TensorRT support

Serving:
- [x] Tensorflow Serving warm up request
- [x] Tensorflow Serving JAVA Client
- [x] Tensorflow Serving Python Client
- [x] Tensorflow Serving Service Control Client
- [x] Tensorflow Serving Server Build and Plugins develop 

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
python main.py --mode=TRAIN --train_dataset_glob=<your dataset glob> --epochs=50 --epochs=50 --mode=TRAIN
```

### Predict:
``` bash
python main.py --mode=IMAGE --model=<your_model_path>
```

### MAP:
``` bash
python main.py --mode=MAP --model=<your_model_path> --test_dataset_glob=<your dataset glob>
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
aeroplane ap:  0.6721874861775297
bicycle ap:  0.7844226664948993
bird ap:  0.6863393529648882
boat ap:  0.5102715372530052
bottle ap:  0.4098093697072679
bus ap:  0.7646277543282962
car ap:  0.8000339732789448
cat ap:  0.8681120849855787
chair ap:  0.4021823009684314
cow ap:  0.6768311030872428
diningtable ap:  0.626045232887253
dog ap:  0.8293983813984888
horse ap:  0.8315961581768014
motorbike ap:  0.771283337747543
person ap:  0.7298645793931624
pottedplant ap:  0.3081565644702266
sheep ap:  0.6510012751038824
sofa ap:  0.6442699680945367
train ap:  0.8025086962000969
tvmonitor ap:  0.6239227675451299
mAP:  0.6696432295131602
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
aeroplane ap:  0.7770436248733187
bicycle ap:  0.822183784348553
bird ap:  0.7346967323068865
boat ap:  0.6142903989882571
bottle ap:  0.4518063126765959
bus ap:  0.782237197681936
car ap:  0.8138978890046222
cat ap:  0.8800232369515162
chair ap:  0.4531520519719176
cow ap: 0.6992367978932157
diningtable ap:  0.6765065569475968
dog ap:  0.8612118810883834
horse ap:  0.8559580684256001
motorbike ap:  0.8027311717682002
person ap:  0.7280218883512792
pottedplant ap:  0.35520418960051925
sheep ap:  0.6833401035128458
sofa ap:  0.6753841073186044
train ap:  0.8107647793504738
tvmonitor ap:  0.6726791558585905
mAP:  0.7075184964459456
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
aeroplane ap:  0.8572154850266848
bicycle ap:  0.8129962658687486
bird ap:  0.8325678324285539
boat ap:  0.7061501348114156
bottle ap:  0.5603823420846883
bus ap:  0.8536452418769342
car ap:  0.8395446870008888
cat ap:  0.9200504816535645
chair ap:  0.514644868267842
cow ap:  0.8202171886452714
diningtable ap:  0.7370149790284737
dog ap:  0.900374518831019
horse ap:  0.8632567146990895
motorbike ap:  0.8147344820261591
person ap:  0.7690434789031615
pottedplant ap:  0.4576271726152926
sheep ap:  0.8006580581981677
sofa ap:  0.7478146395952494
train ap:  0.8783508559769437
tvmonitor ap:  0.6923886096918628
mAP:  0.7689339018615006
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