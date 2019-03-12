# mobilenetv2-yolov3
Tensorflow implementation mobilenetv2-yolov3 inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3.git)

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

# Performance
3 times faster than darknet53-yolov3 with alpha=1.4 and higher accuracy