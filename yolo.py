# -*- coding: utf-8 -*-
"""Class definition of YOLO_v3 style detection model on image and video."""

import colorsys
from timeit import default_timer as timer
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import tensorflow as tf
from yolo3.model import darknet_yolo_body, mobilenetv2_yolo_body, efficientnet_yolo_body, YoloEval
from yolo3.utils import letterbox_image, get_anchors, get_classes
from yolo3.enums import BACKBONE
from yolo3.map import MAPCallback
import os
from typing import List, Tuple
from tensorflow_serving.apis import prediction_log_pb2, predict_pb2
from functools import partial
from tensorflow.python.compiler.tensorrt import trt_convert as trt

tf.keras.backend.set_learning_phase(0)


class YoloModel(tf.keras.Model):
    def __init__(self,
                 model_body,
                 num_anchors,
                 classes,
                 model_path,
                 anchors,
                 input_shape,
                 score=0.2,
                 nms=0.5,
                 with_classes=False,
                 name=None,
                 **kwargs):
        super(YoloModel, self).__init__(name=name, **kwargs)
        self.model_body = model_body
        self.num_anchors = num_anchors
        self.classes = classes
        self.with_classes = with_classes
        self.num_classes = len(classes)
        self.model_path = model_path
        self.anchors = anchors
        self.score = score
        self.nms = nms
        self.input_shapes = input_shape
        self.model = self.model_body(
            tf.keras.layers.Input(
                shape=[*input_shape, 3], batch_size=1, dtype=tf.float32),
            num_anchors=self.num_anchors // 3,
            num_classes=self.num_classes)
        self.model.load_weights(self.model_path)
        self.yolo_eval=YoloEval(
            self.anchors,
            self.num_classes,
            score_threshold=self.score,
            iou_threshold=self.nms,
            name='yolo')

    def parse_image(self, image):
        decoded_image = tf.io.decode_image(image, channels=3, dtype=tf.float32)
        decoded_image.set_shape([None, None, 3])
        letterboxed_image = letterbox_image(decoded_image,
                                               self.input_shapes)
        return decoded_image, letterboxed_image

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1), dtype=tf.string, name='predict_image')
    ])
    def call(self, input):
        decoded_image, input_image = self.parse_image(input[0])
        decoded_image_shape = tf.shape(decoded_image)[0:2]
        input_image = tf.reshape(input_image, [-1, *self.input_shapes, 3])
        input_image = tf.cast(input_image, tf.float32)
        out_boxes, out_scores, out_classes = self.yolo_eval(self.model(input_image),decoded_image_shape)
        if self.with_classes:
            out_classes = tf.gather(self.classes, out_classes)
        return out_boxes, out_scores, out_classes


class YOLO(object):
    def __init__(self, FLAGS):
        self.backbone = FLAGS.get('backbone', BACKBONE.MOBILENETV2)
        self.class_names = get_classes(
            FLAGS.get('classes_path', 'model_data/voc_classes.txt'))
        self.anchors = get_anchors(
            FLAGS.get('anchors_path', 'model_data/yolo_anchors'))
        self.input_shape = FLAGS.get('input_size', (416, 416))
        self.score = FLAGS.get('score', 0.2)
        self.nms = FLAGS.get('nms', 0.5)
        self.with_classes = FLAGS.get('with_classes', False)

        self.generate(FLAGS)

    def generate(self, FLAGS):
        model_path = os.path.expanduser(FLAGS['model'])
        if model_path.endswith('.h5') is not True:
            model_path = tf.train.latest_checkpoint(model_path)

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = tf.keras.models.load_model(
                model_path, compile=False)
        except:
            if self.backbone == BACKBONE.MOBILENETV2:
                model_body = partial(
                    mobilenetv2_yolo_body, alpha=FLAGS.get('alpha', 1.4))
            elif self.backbone == BACKBONE.DARKNET53:
                model_body = darknet_yolo_body
            elif self.backbone == BACKBONE.EFFICIENTNET:
                model_body = partial(
                    efficientnet_yolo_body,
                    num_anchors=num_anchors // 3,
                    num_classes=num_classes,
                    drop_rate=0.2,
                    data_format="channels_last")

            self.yolo_model = YoloModel(
                model_body, num_anchors, self.class_names, model_path,
                self.anchors, self.input_shape, self.score, self.nms,
                self.with_classes)

        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(model_path))
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

    def detect_image(self, image, draw=True):
        image_data = image
        if isinstance(image, bytes) is False:
            image_data = image.read()
        start = timer()
        out_boxes, out_scores, out_classes = self.yolo_model([image_data])
        if tf.executing_eagerly():
            out_boxes = out_boxes.numpy()
            out_scores = out_scores.numpy()
            out_classes = out_classes.numpy()
        else:
            start = timer()
            out_boxes, out_scores, out_classes = tf.compat.v1.keras.backend.get_session(
            ).run([out_boxes, out_scores, out_classes])
        end = timer()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        if draw:
            image = Image.open(image)
            font = ImageFont.truetype(
                font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[1] + image.size[0]) // 300
            draw = ImageDraw.Draw(image)
            for i, c in reversed(list(enumerate(out_classes))):
                if self.with_classes:
                    c = self.class_names.index(str(c, encoding="utf-8"))
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


def overwrite_path(path):
    if tf.io.gfile.exists(path):
        while True:
            overwrite = input("Overwrite existed model(yes/no):")
            if overwrite == 'yes':
                tf.io.gfile.rmtree(path)
                break
            elif overwrite == 'no':
                raise ValueError(
                    "Export directory already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory: "
                    + path)
            else:
                print('Please input yes/no')


def export_tfjs_model(yolo, path):
    import tensorflowjs as tfjs
    import tempfile
    overwrite_path(path)

    temp_savedmodel_dir = tempfile.mktemp(suffix='.savedmodel')
    tf.keras.experimental.export_saved_model(
        yolo.yolo_model, temp_savedmodel_dir, serving_only=True)

    tfjs.converters.tf_saved_model_conversion_v2.convert_tf_saved_model(
        temp_savedmodel_dir,
        path,
        signature_def='serving_default',
        saved_model_tags='serve')
    # tfjs.converters.save_keras_model(yolo.yolo_model,
    #                                  path)


def export_serving_model(yolo, path, warmup_path=None,with_tensorrt=False):
    overwrite_path(path)
    tf.saved_model.save(yolo.yolo_model, path)
    if with_tensorrt:
        params=trt.TrtConversionParams(
            rewriter_config_template=None,
            max_workspace_size_bytes=trt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
            precision_mode=trt.TrtPrecisionMode.FP16,
            minimum_segment_size=3,
            is_dynamic_op=True,
            maximum_cached_engines=1,
            use_calibration=True,
            max_batch_size=1)
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=path,conversion_params=params)
        converter.convert()
        tf.io.gfile.rmtree(path)
        converter.save(path)
    asset_extra = os.path.join(path, "assets.extra")
    tf.io.gfile.mkdir(asset_extra)
    with tf.io.TFRecordWriter(
            os.path.join(asset_extra, "tf_serving_warmup_requests")) as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
        if warmup_path is None:
            warmup_path = input('Please enter warm up image path:')
        image = open(warmup_path, 'rb').read()
        image_data = np.expand_dims(image, 0)
        request.inputs['predict_image'].CopyFrom(
            tf.compat.v1.make_tensor_proto(image_data))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


def export_tflite_model(yolo, path):
    overwrite_path(path)
    converter = tf.lite.TFLiteConverter.from_keras_model(yolo.yolo_model)
    # converter.allow_custom_ops = True
    # converter.experimental_enable_mlir_converter=True
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    tf.io.write_file(os.path.join(path,'model.tflite'),tflite_model)


def calculate_map(yolo, glob):
    mAP = MAPCallback(glob, yolo.input_shape, yolo.class_names)
    mAP.set_model(yolo.yolo_model)
    APs = mAP.calculate_aps()
    for cls in range(len(yolo.class_names)):
        if cls in APs:
            print(yolo.class_names[cls] + ' ap: ', APs[cls])
    mAP = np.mean([APs[cls] for cls in APs])
    print('mAP: ', mAP)


def inference_img(yolo, image_path, draw=True):
    try:
        image = open(image_path, 'rb')
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image, draw)
        r_image.show()


def detect_img(yolo):
    while True:
        inputs = input('Input image filename:')
        if inputs.endswith('.txt'):
            with open(input) as file:
                for image_path in file.readlines():
                    image_path = image_path.strip()
                    inference_img(yolo, image_path, False)
        else:
            inference_img(yolo, inputs)
    yolo.close_session()


def detect_video(yolo: YOLO, video_path: str, output_path: str = ""):
    video_path_formatted = video_path
    if video_path.isdigit():
        video_path_formatted = int(video_path)
    vid = cv2.VideoCapture(video_path_formatted)

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

    trackers = {}
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=30)
    thickness = 1
    frame_count = 0
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        img_str = cv2.imencode('.jpg', np.array(image))[1].tostring()
        draw = ImageDraw.Draw(image)
        if len(trackers) > 0:
            for tracker in trackers:
                success, box = tracker.update(frame)
                if success is not True:
                    trackers.pop(tracker)
                    continue
                left, top, width, height = box
                right = left + width
                bottom = top + height

                label = '{}'.format(trackers[tracker])

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
                    trackers = {}
                    frame_count = 0
        else:
            boxes, scores, classes = yolo.detect_image(img_str, False)
            for i, c in enumerate(classes):
                predicted_class = yolo.class_names[c]
                top, left, bottom, right = boxes[i]
                height = abs(bottom - top)
                width = abs(right - left)
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                #tracker = cv2.TrackerMOSSE_create()
                tracker.init(frame, (left, top, width, height))
                trackers[tracker] = predicted_class

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
        cv2.putText(
            result,
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
