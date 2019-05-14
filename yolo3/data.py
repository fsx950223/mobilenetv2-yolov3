import tensorflow as tf
from functools import reduce
from yolo3.utils import get_random_data
from yolo3.model import preprocess_true_boxes
from yolo3.enum import DATASET_MODE
from random import random
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if hasattr(self, 'input_shapes'):
            index = math.floor(random() * len(self.input_shapes))
            self.input_shape.assign(self.input_shapes[index])
    def parse_tfrecord(self,example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_image(features['image/encoded'], channels=3, dtype=tf.float32)
        image.set_shape([None, None, 3])
        xmins = features['image/object/bbox/xmin'].values
        xmaxs = features['image/object/bbox/xmax'].values
        ymins = features['image/object/bbox/ymin'].values
        ymaxs = features['image/object/bbox/ymax'].values
        labels = features['image/object/bbox/label'].values
        image, bbox = get_random_data(image, xmins, xmaxs, ymins, ymaxs, labels, self.input_shape, train=self.mode==DATASET_MODE.TRAIN)
        y1, y2, y3 = tf.py_function(preprocess_true_boxes, [bbox, self.input_shape, self.anchors, self.num_classes],
                                    [tf.float32, tf.float32, tf.float32])
        return image, (y1, y2, y3)

    def tfrecord_dataset(self, files):
        # with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        """data generator for fit_generator"""

        if self.mode == DATASET_MODE.TRAIN:
            train_num = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).shuffle(train_num).map(self.parse_tfrecord,
                                                                                             num_parallel_calls=AUTOTUNE).prefetch(
                self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.VALIDATE:
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).map(self.parse_tfrecord, num_parallel_calls=AUTOTUNE).prefetch(
                self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.TEST:
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).map(self.parse_tfrecord,num_parallel_calls=AUTOTUNE).prefetch(
                self.batch_size).batch(self.batch_size)
        return dataset

    def parse_text(self,line):
        values = tf.strings.split([line]).values
        image = tf.image.decode_image(tf.io.read_file(values[0]), channels=3, dtype=tf.float32)
        image.set_shape([None, None, 3])
        reshaped_data = tf.reshape(values[1:], [-1, 5])
        xmin = tf.strings.to_number(reshaped_data[:, 0], tf.float32)
        xmax = tf.strings.to_number(reshaped_data[:, 1], tf.float32)
        ymin = tf.strings.to_number(reshaped_data[:, 2], tf.float32)
        ymax = tf.strings.to_number(reshaped_data[:, 3], tf.float32)
        label = tf.strings.to_number(reshaped_data[:, 4], tf.float32)
        image, bbox = get_random_data(image, xmin, xmax, ymin, ymax, label, self.input_shape, train=self.mode==DATASET_MODE.TRAIN)
        y1, y2, y3 = tf.py_function(preprocess_true_boxes, [bbox, self.input_shape, self.anchors, self.num_classes],
                                    [tf.float32, tf.float32, tf.float32])
        return image, (y1, y2, y3)

    def text_dataset(self, files):
        # with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(files)



        if self.mode==DATASET_MODE.TRAIN:
            train_sum = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).shuffle(train_sum).map(self.parse_text,
                                                                                             num_parallel_calls=AUTOTUNE).batch(
                self.batch_size).prefetch(self.batch_size).repeat()
        elif self.mode==DATASET_MODE.VALIDATE:
            dataset = dataset.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).map(self.parse_text, num_parallel_calls=AUTOTUNE).batch(
                self.batch_size).prefetch(self.batch_size).repeat()
        elif self.mode==DATASET_MODE.TEST:
            dataset = dataset.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=len(files), num_parallel_calls=AUTOTUNE).map(self.parse_text,
                                                                          num_parallel_calls=AUTOTUNE).batch(
                self.batch_size).prefetch(self.batch_size)
        return dataset

    def __init__(self, glob_path: str, batch_size,anchors=None,num_classes=None,input_shapes=None, mode=DATASET_MODE.TRAIN):
        self.glob_path = glob_path
        self.batch_size = batch_size
        if isinstance(input_shapes, list):
            self.input_shapes = input_shapes
            self.input_shape = tf.Variable(name="input_shape", initial_value=self.input_shapes[0], trainable=False)
        else:
            self.input_shape = input_shapes
        self.anchors = anchors
        self.num_classes = num_classes
        self.mode = mode

    def _get_num_from_name(self, name):
        return int(name.split('/')[-1].split('.')[0].split('_')[-1])

    def build(self):
        files = tf.io.gfile.glob(self.glob_path)
        try:
            num = reduce(lambda x, y: x + y, map(lambda file: self._get_num_from_name(file), files))
        except Exception:
            raise ValueError('Please format file name like <name>_<number>.<extension>')
        else:
            if self.glob_path.endswith('.tfrecords'):
                return self.tfrecord_dataset(files), num
            elif self.glob_path.endswith('.txt'):
                return self.text_dataset(files), num
