# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import tensorflow as tf
from yolo3.model import yolo_eval, darknet_yolo_body, mobilenetv2_yolo_body, inception_yolo_body, densenet_yolo_body
from yolo3.utils import letterbox_image, get_anchors, get_classes
from yolo3.enum import OPT, BACKBONE
from yolo3.map import MAPCallback
import os
from typing import List, Tuple
from tensorflow_serving.apis import prediction_log_pb2,predict_pb2
from tensorflow.python import debug as tf_debug

# if hasattr(tf, 'enable_eager_execution'):
#     tf.enable_eager_execution()
tf.keras.backend.set_learning_phase(0)
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_num = len(gpus.split(','))


class YOLO(object):
    _defaults = {
        "backbone": BACKBONE.MOBILENETV2,
        "model_config": {
            BACKBONE.MOBILENETV2: {
                "input_size": (416, 416),
                "model_path":'../download/mobilenetv2_trained_weights_final (5).h5',
                #"anchors_path": 'model_data/yolo_anchors.txt',
                #"classes_path": 'model_data/voc_classes.txt',
                # "model_path":"./logs/mobilenetv22019-05-24/ep009-loss19.673-val_loss19.141.h5",
                "anchors_path": 'model_data/cci_anchors.txt',
                "classes_path": 'model_data/cci.names',
                "alpha": 1.4
            },
            BACKBONE.DARKNET53: {
                "input_size": (416, 416),
                "model_path": '../download/darknet53_trained_weights_final.h5',
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
        "score": 0.2,
        "nms": 0.5,
        "opt": OPT.XLA
    }

    @classmethod
    def get_defaults(cls, n: str):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        #self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = get_classes(
            self.model_config[self.backbone]['classes_path'])
        self.anchors = get_anchors(
            self.model_config[self.backbone]['anchors_path'])
        self.input_shape = self.model_config[self.backbone]['input_size']
        config = tf.ConfigProto()

        if self.opt == OPT.XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)
        elif self.opt == OPT.MKL:
            config.intra_op_parallelism_threads = 4
            config.inter_op_parallelism_threads = 4
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)
        elif self.opt == OPT.DEBUG:
            tf.logging.set_verbosity(tf.logging.DEBUG)
            sess = tf_debug.TensorBoardDebugWrapperSession(
                tf.Session(config=tf.ConfigProto(log_device_placement=True)),
                "localhost:6064")
            tf.keras.backend.set_session(sess)
        else:
            sess = tf.get_session()
        self.sess = sess
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(
            self.model_config[self.backbone]['model_path'])
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except:
            if self.backbone == BACKBONE.MOBILENETV2:
                if tf.executing_eagerly():
                    model = mobilenetv2_yolo_body(
                        tf.keras.layers.Input(shape=(*self.input_shape, 3),
                                              name='predict_image'),
                        num_anchors // 3, num_classes, self.model_config[self.backbone]['alpha'])
                else:
                    self.input = tf.keras.layers.Input(shape=(None, None, 3),
                                                       name='predict_image',
                                                       dtype=tf.uint8)
                    input = tf.map_fn(
                        lambda image: tf.image.convert_image_dtype(
                            image, tf.float32), self.input, tf.float32)
                    image, shape = letterbox_image(input, self.input_shape)
                    self.input_image_shape = tf.shape(input)[1:3]
                    image = tf.reshape(image, [-1, *self.input_shape, 3])
                    model = mobilenetv2_yolo_body(image, num_anchors // 3,
                                                  num_classes, self.model_config[self.backbone]['alpha'])
            elif self.backbone == BACKBONE.DARKNET53:
                model = darknet_yolo_body(
                    tf.keras.layers.Input(shape=(*self.input_shape, 3),
                                          name='predict_image'),
                    num_anchors // 3, num_classes)
            elif self.backbone == BACKBONE.DENSENET:
                model = densenet_yolo_body(
                    tf.keras.layers.Input(shape=(None, None, 3)),
                    num_anchors // 3, num_classes)
            elif self.backbone == BACKBONE.INCEPTION_RESNET2:
                model = inception_yolo_body(
                    tf.keras.layers.Input(shape=(None, None, 3)),
                    num_anchors // 3, num_classes)
            model.load_weights(
                model_path)  # make sure model, anchors and classes match
        else:
            assert model.layers[-1].output_shape[-1] == \
                   num_anchors / len(model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        hsv_tuples: List[Tuple[float, float, float]] = [
            (x / len(self.class_names), 1., 1.)
            for x in range(len(self.class_names))
        ]
        self.colors: List[Tuple[float, float, float]] = list(
            map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors: List[Tuple[int, int, int]] = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
            self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        if tf.executing_eagerly():
            self.yolo_model = model
        else:
            output = tf.keras.layers.Lambda(lambda input: yolo_eval(
                input,
                self.anchors,
                len(self.class_names),
                self.input_image_shape,
                score_threshold=self.score,
                iou_threshold=self.nms),name='yolo')(model.output)
            self.yolo_model = tf.keras.Model(model.input, output)
        # Generate output tensor targets for filtered bounding boxes.
        if gpu_num >= 2:
            self.yolo_model = tf.keras.utils.multi_gpu_model(self.yolo_model,
                                                             gpus=gpu_num)

    def export_serving_model(self, path: str) -> None:
        if tf.version.VERSION == '1.13.1':
            tf.saved_model.simple_save(
                self.sess,
                path,
                inputs={'predict_image:0': self.input},
                outputs={t.name: t for t in self.yolo_model.output})
        else:
            signature_def_map = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.predict_signature_def(
                    {'predict_image:0': self.input},
                    {t.name: t for t in self.yolo_model.output})
            }
            tf.saved_model.experimental.save(self.yolo_model, path,
                                             signature_def_map)
        asset_extra=os.path.join(path,"assets.extra")
        tf.io.gfile.mkdir(asset_extra)
        with tf.io.TFRecordWriter(os.path.join(asset_extra,"tf_serving_warmup_requests")) as writer:
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'detection'
            request.model_spec.signature_name = 'serving_default'
            image = Image.open('../download/image3.jpeg')
            scale = self.input_shape[0] / max(image.size)
            if scale < 1:
                image = image.resize((int(line * scale) for line in image.size),
                                     Image.BILINEAR)
            image_data = np.array(image, dtype='uint8')
            image_data = np.expand_dims(image_data, 0)
            request.inputs['predict_image:0'].CopyFrom(tf.make_tensor_proto(image_data))
            log=prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())

    def export_tflite_model(self, path: str) -> None:
        converter = tf.lite.TFLiteConverter.from_session(
            self.sess, [self.input], [self.boxes, self.scores, self.classes])
        converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
        tflite_model = converter.convert()
        open(path, "wb").write(tflite_model)

    def detect_image(self, image,draw=True) -> Image:
        if tf.executing_eagerly():
            image_data = tf.expand_dims(image, 0)
            if self.input_shape != (None, None):
                assert self.input_shape[0] % 32 == 0, 'Multiples of 32 required'
                assert self.input_shape[1] % 32 == 0, 'Multiples of 32 required'
                boxed_image, image_shape = letterbox_image(
                    image_data, tuple(reversed(self.input_shape)))
            else:
                height, width, _ = image_data.shape
                new_image_size = (width - (width % 32), height - (height % 32))
                boxed_image, image_shape = letterbox_image(
                    image_data, new_image_size)
            image_data = np.array(boxed_image)
            start = timer()
            output = self.yolo_model.predict(image_data)
            out_boxes, out_scores, out_classes = yolo_eval(
                output,
                self.anchors,
                len(self.class_names),
                image.shape[0:2],
                score_threshold=self.score,
                iou_threshold=self.nms)
            end = timer()
            image = Image.fromarray((np.array(image) * 255).astype('uint8'),
                                    'RGB')
        else:
            image_data = np.expand_dims(image, 0)
            start = timer()
            out_boxes, out_scores, out_classes = self.sess.run(
                self.yolo_model.output, feed_dict={self.input: image_data})
            end = timer()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        if draw:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] +
                                                    0.5).astype('int32'))
            thickness = (image.size[1] + image.size[0]) // 300
            draw = ImageDraw.Draw(image)
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)

                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],
                                   outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin),
                     tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            print(end - start)
            return image
        else:
            return out_boxes, out_scores, out_classes

    def calculate_map(self, glob):

        def parse_fn(example_proto):
            feature_description = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
            }
            features = tf.io.parse_single_example(example_proto,
                                                  feature_description)
            image = tf.image.decode_image(features['image/encoded'],
                                          channels=3,
                                          dtype=tf.float32)
            xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
            xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
            ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
            ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
            label = tf.expand_dims(features['image/object/bbox/label'].values,
                                   0)
            bbox = tf.concat(
                [xmin, ymin, xmax, ymax,
                 tf.cast(label, tf.float32)], 0)
            return image, bbox

        mAP = MAPCallback(glob,
                          self.input_shape,
                          self.anchors,
                          self.class_names,
                          parse_fn,
                          score=0,
                          batch_size=4)
        mAP.set_model(self.yolo_model)
        APs = mAP.calculate_aps()
        for cls in range(len(self.class_names)):
            if cls in APs:
                print(self.class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)

    def close_session(self):
        self.sess.close()

def detect_imgs(yolo,input):
    with open(input) as file:
        for image_path in file.readlines():
            image_path=image_path.strip()
            try:
                if tf.executing_eagerly():
                    content = tf.io.read_file(image_path)
                    image = tf.image.decode_image(content,
                                                  channels=3,
                                                  dtype=tf.float32)
                else:
                    image = Image.open(image_path)
            except:
                print('Open Error! Try again!')
            else:
                r_image = yolo.detect_image(image)
                r_image.save(os.path.join('chengyun',image_path.split('/')[-1]))
    yolo.close_session()

def detect_img(yolo):
    while True:
        image_path = input('Input image filename:')
        try:
            if tf.executing_eagerly():
                content = tf.io.read_file(image_path)
                image = tf.image.decode_image(content,
                                              channels=3,
                                              dtype=tf.float32)
            else:
                image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def detect_video(yolo: YOLO, video_path: str, output_path: str = ""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC),
              type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    detected=False
    trackers=[]
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=30)
    thickness = 1
    frame_count = 0
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image_data = np.array(image) / 255.
        draw = ImageDraw.Draw(image)
        if detected:
            for tracker,predicted_class in trackers:
                success,box=tracker.update(frame)
                left,top,width,height=box
                right=left+width
                bottom=top+height

                label = '{}'.format(predicted_class)

                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],
                                   outline=yolo.colors[c])
                draw.rectangle(
                    [tuple(text_origin),
                     tuple(text_origin + label_size)],
                    fill=yolo.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                frame_count+=1
                if frame_count == 100:
                    for tracker in trackers:
                        del tracker
                    trackers=[]
                    frame_count=0
                    detected=False
        else:
            if tf.executing_eagerly():
                boxes, scores, classes = yolo.detect_image(image_data,False)
            else:
                boxes, scores, classes = yolo.detect_image(image, False)
            for i, c in enumerate(classes):
                predicted_class = yolo.class_names[c]
                top, left, bottom, right=boxes[i]
                height=abs(bottom-top)
                width=abs(right-left)
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                #tracker = cv2.TrackerMOSSE_create()
                tracker.init(frame,(left,top,width,height))
                trackers.append([tracker,predicted_class])

                label = '{}'.format(predicted_class)
                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],
                                   outline=yolo.colors[c])
                draw.rectangle(
                    [tuple(text_origin),
                     tuple(text_origin + label_size)],
                    fill=yolo.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            detected=True
        del draw
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
        cv2.putText(result,
                    text=fps,
                    org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=(255, 0, 0),
                    thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
