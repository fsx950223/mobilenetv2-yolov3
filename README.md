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
- [ ] IOU
- [x] GIOU

Tensorflow:
- [x] Tensorflow2 Ready
- [x] Use tf.data to load dataset
- [ ] Remove image shape input when use session
- [ ] Convert model to tensorflow lite model
- [ ] Multi GPU training support

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
    ``` python
    opt = <your session config>
    backbone = <your yolov3 backbone>
    log_dir = <path/to/your/tensorboard/log>
    batch_size = <you batch size>
    train_dataset_path = <path/to/your/train/folder>
    val_dataset_path = <path/to/your/val/folder>
    train_dataset_glob = <train glob>
    val_dataset_glob = <val glob>
    ```
    
# Performance
3 times faster than darknet53-yolov3 with alpha=1.4 and higher accuracy

# Pascal Dataset
I have packaged a pascal tfrecords for you.See [here](https://drive.google.com/drive/folders/172sH75LPeUd2yyzAnrce0LLe2UR_kFqF)