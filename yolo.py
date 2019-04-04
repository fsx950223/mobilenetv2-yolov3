# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
from yolo3.model import yolo_eval, darknet_yolo_body, mobilenetv2_yolo_body,inception_yolo_body,densenet_yolo_body
from yolo3.utils import letterbox_image
from yolo3.enum import OPT,BACKBONE
import os
from typing import List, Tuple
from tensorflow.python import debug as tf_debug

if hasattr(tf,'enable_eager_execution'):
    tf.enable_eager_execution()
gpus=""
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_num=len(gpus.split(','))

class YOLO(object):
    _defaults = {
        "backbone":BACKBONE.MOBILENETV2,
        "model_config":{
            BACKBONE.MOBILENETV2:{
                "input_size":(224,224),
                "model_path": '../download/mobilenetv2_trained_weights_stage_12.h5',
                "anchors_path":'model_data/yolo_anchors.txt',
                "classes_path":'model_data/voc_classes.txt'
            },
            BACKBONE.DARKNET53:{
                "input_size":(416,416),
                "model_path": '../download/trained_weights_final6.h5',
                "anchors_path": 'model_data/yolo_anchors.txt',
                "classes_path": 'model_data/voc_classes.txt'
            },
            BACKBONE.INCEPTION_RESNET2: {
                "input_size": (608, 608),
                "model_path": '../download/trained_weights_final6.h5',
                "anchors_path": 'model_data/yolo_anchors.txt',
                "classes_path": 'model_data/voc_classes.txt'
            },
            BACKBONE.DENSENET: {
                "input_size": (416, 416),
                "model_path": '../download/densenet_trained_weights_stage_1.h5',
                "anchors_path": 'model_data/yolo_anchors.txt',
                "classes_path": 'model_data/voc_classes.txt'
            }
        },
        "score": 0.1,
        "iou": 0.5,
        "opt":OPT.XLA
    }

    @classmethod
    def get_defaults(cls, n: str):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.alpha=1.4
        config = tf.ConfigProto()
        tf.keras.backend.set_learning_phase(0)
        if self.opt==OPT.XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)
        elif self.opt==OPT.MKL:
            config.intra_op_parallelism_threads = 4
            config.inter_op_parallelism_threads = 4
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)
        elif self.opt==OPT.DEBUG:
            sess=tf_debug.TensorBoardDebugWrapperSession(tf.get_session(), "fangsixie-Inspiron-7572:6064")
            tf.keras.backend.set_session(sess)
        else:
            sess = tf.get_session()
        self.sess = sess
        if tf.executing_eagerly():
            self.generate()
        else:
            self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self) -> List[str]:
        classes_path = os.path.expanduser(self.model_config[self.backbone]['classes_path'])
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self) -> np.ndarray:
        anchors_path = os.path.expanduser(self.model_config[self.backbone]['anchors_path'])
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_config[self.backbone]['model_path'])
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = tf.keras.models.load_model(model_path, compile=False)
        except:
            if self.backbone==BACKBONE.MOBILENETV2:
                self.yolo_model = mobilenetv2_yolo_body(
                    tf.keras.layers.Input(shape=(*self.model_config[self.backbone]['input_size'], 3), name='predict_image'),
                    num_anchors // 3,
                    num_classes, self.alpha)
            elif self.backbone==BACKBONE.DARKNET53:
                self.yolo_model = darknet_yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            elif self.backbone == BACKBONE.DENSENET:
                self.yolo_model = densenet_yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 3,
                                                    num_classes)
            elif self.backbone==BACKBONE.INCEPTION_RESNET2:
                self.yolo_model = inception_yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        hsv_tuples: List[Tuple[float, float, float]] = [(x / len(self.class_names), 1., 1.)
                                                        for x in range(len(self.class_names))]
        self.colors: List[Tuple[float, float, float]] = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors: List[Tuple[int, int, int]] = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        if gpu_num >= 2:
            self.yolo_model = tf.keras.utils.multi_gpu_model(self.yolo_model, gpus=gpu_num)
        if tf.executing_eagerly() is not True:
            self.input_image_shape = tf.placeholder(tf.float32, shape=(2,),name="image_size")
            boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                               len(self.class_names), self.input_image_shape,
                                               score_threshold=self.score, iou_threshold=self.iou)
            return boxes, scores, classes

    def export_serving_model(self, path: str) -> None:
        tf.saved_model.simple_save(
            self.sess,
            path,
            inputs={
                'predict_image:0': self.yolo_model.input,
                'image_size:0': self.input_image_shape
            },
            outputs={t.name: t for t in [self.boxes, self.scores, self.classes]})

    def detect_image(self, image) -> Image:
        if tf.executing_eagerly():
            if self.model_config[self.backbone]['input_size'] != (None, None):
                assert self.model_config[self.backbone]['input_size'][0] % 32 == 0, 'Multiples of 32 required'
                assert self.model_config[self.backbone]['input_size'][1] % 32 == 0, 'Multiples of 32 required'
                boxed_image, image_shape = letterbox_image(image, tuple(
                    reversed(self.model_config[self.backbone]['input_size'])))
            else:
                height, width, _ = image.shape
                new_image_size = (width - (width % 32), height - (height % 32))
                boxed_image, image_shape = letterbox_image(image, new_image_size)
            image_data = np.expand_dims(boxed_image, 0)
            start = timer()
            output=self.yolo_model.predict(image_data)
            out_boxes, out_scores, out_classes=yolo_eval(output,self.anchors,len(self.class_names),image_shape[0:2],
                    score_threshold=self.score, iou_threshold=self.iou)
            end = timer()
            image = Image.fromarray((np.array(image) * 255).astype('uint8'), 'RGB')
        else:
            size=self.model_config[self.backbone]['input_size']
            iw, ih = image.size
            w, h = size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            resized_image = image.resize((nw, nh), Image.BILINEAR)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(resized_image, ((w - nw) // 2, (h - nh) // 2))
            image_data=np.array(new_image,dtype='float32')
            image_data/=255.
            image_data=np.expand_dims(image_data,0)
            start = timer()
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    "predict_image:0": image_data,
                    "image_size:0":tuple(reversed(resized_image.size))
                })
            end = timer()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[1] + image.size[0]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right =box*(tuple(reversed(image.size))*2)
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        print(end - start)
        return image

    def detect_image_test(self, image):

        if self.model_config[self.backbone]['input_size'] != (None, None):
            assert self.model_config[self.backbone]['input_size'][0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_config[self.backbone]['input_size'][1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_config[self.backbone]['input_size'])))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]]
            })

        return out_boxes, out_scores, reversed(list(enumerate(out_classes)))

    def close_session(self):
        self.sess.close()


def detect_video(yolo: YOLO, video_path: str, output_path: str = ""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        if tf.executing_eagerly():
            image=np.array(image) / 255.
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
