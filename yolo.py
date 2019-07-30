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
from yolo3.model import yolo_eval, darknet_yolo_body, mobilenetv2_yolo_body, efficientnet_yolo_body, YoloEval
from yolo3.utils import letterbox_image, get_anchors, get_classes
from yolo3.enum import OPT, BACKBONE
from yolo3.map import MAPCallback
import os
from typing import List, Tuple
from tensorflow_serving.apis import prediction_log_pb2, predict_pb2
from tensorflow.python import debug as tf_debug
from functools import partial

tf.keras.backend.set_learning_phase(0)


class YOLO(object):
    _defaults = {
        "score": 0.2,
        "nms": 0.5,
    }

    @classmethod
    def get_defaults(cls, n: str):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, FLAGS):
        self.__dict__.update(self._defaults)  # set up default values
        self.backbone = FLAGS['backbone']
        self.opt = FLAGS['opt']
        self.class_names = get_classes(FLAGS['classes_path'])
        self.anchors = get_anchors(FLAGS['anchors_path'])
        self.input_shape = FLAGS['input_size']
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
            sess = tf.keras.backend.get_session()
        self.sess = sess
        self.generate(FLAGS)

    def generate(self, FLAGS):
        model_path = os.path.expanduser(FLAGS['model'])
        if model_path.endswith(
            '.h5') is not True:
            model_path=tf.train.latest_checkpoint(model_path)

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except:
            if self.backbone == BACKBONE.MOBILENETV2:
                model_body = partial(mobilenetv2_yolo_body,
                                     alpha=FLAGS['alpha'])
            elif self.backbone == BACKBONE.DARKNET53:
                model_body = darknet_yolo_body
            elif self.backbone == BACKBONE.EFFICIENTNET:
                model_body = partial(efficientnet_yolo_body,
                                     model_name='efficientnet-b4')
            if tf.executing_eagerly():
                input = tf.keras.layers.Input(shape=(*self.input_shape, 3),
                                              name='predict_image')
                model = model_body(input,
                                   num_anchors=num_anchors // 3,
                                   num_classes=num_classes)
            else:
                input = tf.keras.layers.Input(shape=(None, None, 3),
                                              name='predict_image',
                                              dtype=tf.uint8)
                input_image = tf.map_fn(
                    lambda image: tf.image.convert_image_dtype(
                        image, tf.float32), input, tf.float32)
                image, shape = letterbox_image(input_image, self.input_shape)
                self.input_image_shape = tf.shape(input_image)[1:3]
                image = tf.reshape(image, [-1, *self.input_shape, 3])
                model = model_body(image,
                                   num_anchors=num_anchors // 3,
                                   num_classes=num_classes)
            self.input = input
            model.load_weights(
                model_path)  # make sure model, anchors and classes match
        else:
            assert model.layers[-1].output_shape[-1] == \
                   num_anchors / len(model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        if tf.executing_eagerly():
            self.yolo_model = model
        else:
            output = YoloEval(self.anchors,
                              len(self.class_names),
                              self.input_image_shape,
                              score_threshold=self.score,
                              iou_threshold=self.nms,
                              name='yolo')(model.output)
            self.yolo_model = tf.keras.Model(model.input, output)
        # Generate output tensor targets for filtered bounding boxes.
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

    def detect_image(self, image, draw=True) -> Image:
        if tf.executing_eagerly():
            image_data = tf.expand_dims(image, 0)
            if self.input_shape != (None, None):
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


def export_tfjs_model(yolo, path):
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(yolo.yolo_model,
                                     path,
                                     quantization_dtype=np.uint8)


def export_serving_model(yolo, path):
    if tf.io.gfile.exists(path):
        overwrite = input("Overwrite existed model(yes/no):")
        if overwrite == 'yes':
            tf.io.gfile.rmtree(path)
        else:
            raise ValueError(
                "Export directory already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory: "
                + path)
    tf.saved_model.simple_save(
        yolo.sess,
        path,
        inputs={'predict_image:0': yolo.input},
        outputs={t.name: t for t in yolo.yolo_model.output})

    asset_extra = os.path.join(path, "assets.extra")
    tf.io.gfile.mkdir(asset_extra)
    with tf.io.TFRecordWriter(
            os.path.join(asset_extra, "tf_serving_warmup_requests")) as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
        image = Image.open('../download/image3.jpeg')
        scale = yolo.input_shape[0] / max(image.size)
        if scale < 1:
            image = image.resize((int(line * scale) for line in image.size),
                                 Image.BILINEAR)
        image_data = np.array(image, dtype='uint8')
        image_data = np.expand_dims(image_data, 0)
        request.inputs['predict_image:0'].CopyFrom(
            tf.make_tensor_proto(image_data))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


def export_tflite_model(yolo, path):
    yolo.yolo_model.input.set_shape([None, *yolo.input_shape, 3])
    converter = tf.lite.TFLiteConverter.from_session(
        yolo.sess, [yolo.yolo_model.input], list(yolo.yolo_model.output))
    converter.allow_custom_ops = True
    converter.inference_type = tf.lite.constants.FLOAT
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    tflite_model = converter.convert()
    tf.io.gfile.GFile(path, "wb").write(tflite_model)


def calculate_map(yolo, glob):
    mAP = MAPCallback(glob, yolo.input_shape, yolo.anchors, yolo.class_names)
    mAP.set_model(yolo.yolo_model)
    APs = mAP.calculate_aps()
    for cls in range(len(yolo.class_names)):
        if cls in APs:
            print(yolo.class_names[cls] + ' ap: ', APs[cls])
    mAP = np.mean([APs[cls] for cls in APs])
    print('mAP: ', mAP)

def inference_img(image_path):
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


def detect_img(yolo):
    while True:
        inputs = input('Input image filename:')
        if inputs.endsWith('.txt'):
            with open(input) as file:
                for image_path in file.readlines():
                    image_path = image_path.strip()
                    inference_img(image_path)
        else:
            inference_img(inputs)
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
    detected = False
    trackers = []
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=30)
    thickness = 1
    frame_count = 0
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image_data = np.array(image) / 255.
        draw = ImageDraw.Draw(image)
        if detected:
            for tracker, predicted_class in trackers:
                success, box = tracker.update(frame)
                left, top, width, height = box
                right = left + width
                bottom = top + height

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
                frame_count += 1
                if frame_count == 100:
                    for tracker in trackers:
                        del tracker
                    trackers = []
                    frame_count = 0
                    detected = False
        else:
            if tf.executing_eagerly():
                boxes, scores, classes = yolo.detect_image(image_data, False)
            else:
                boxes, scores, classes = yolo.detect_image(image, False)
            for i, c in enumerate(classes):
                predicted_class = yolo.class_names[c]
                top, left, bottom, right = boxes[i]
                height = abs(bottom - top)
                width = abs(right - left)
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                #tracker = cv2.TrackerMOSSE_create()
                tracker.init(frame, (left, top, width, height))
                trackers.append([tracker, predicted_class])

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
            detected = True
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
