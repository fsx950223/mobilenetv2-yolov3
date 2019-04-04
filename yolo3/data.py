import tensorflow as tf
from yolo3.model import preprocess_true_boxes
from typing import Tuple, List
from functools import reduce
from yolo3.utils import get_random_data
import os
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
def draw_image(dataset):
    file_writer_cm = tf.summary.create_file_writer('logs/cm')
    iter=dataset.__iter__()
    images,_,_,_=iter.get_next()[0]
    images = np.reshape(images[0:25], (-1, 28, 28, 1))
    with file_writer_cm.as_default():
        tf.summary.image('input_image', images,max_outputs=25, step=0)

def tfrecord_dataset(files: List[str], batch_size: int, input_shape: Tuple[int, int],
                   anchors: List[float],
                   num_classes: int,
                   train: bool = True):
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        """data generator for fit_generator"""

        def parse(example_proto):
            feature_description = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
            }
            features = tf.io.parse_single_example(example_proto, feature_description)
            image, bbox = get_random_data(features['image/encoded'],
                                          features['image/object/bbox/xmin'].values,
                                          features['image/object/bbox/xmax'].values,
                                          features['image/object/bbox/ymin'].values,
                                          features['image/object/bbox/ymax'].values,
                                          features['image/object/bbox/label'].values,
                                          input_shape, train=train)

            y0, y1, y2 = tf.py_function(preprocess_true_boxes, [bbox, input_shape, anchors, num_classes],
                                        [tf.float32, tf.float32, tf.float32])
            return (image, y0, y1, y2), 0

        if train:
            train_num = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).shuffle(train_num).map(parse, num_parallel_calls=AUTOTUNE).repeat().prefetch(batch_size).batch(batch_size)
        else:
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).map(parse, num_parallel_calls=AUTOTUNE).repeat().prefetch(batch_size).batch(batch_size)
        #draw_image(dataset)
        return dataset

def text_dataset(files: List[str], batch_size: int, input_shape: Tuple[int, int],
                   anchors: List[float],
                   num_classes: int,
                   train: bool = True):
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        def parse(line):
            values=tf.strings.split([line]).values
            def get_data(values):
                i=0;
                xmin=[]
                ymin=[]
                xmax=[]
                ymax=[]
                label=[]
                while i<len(values):
                    xmin.append(values[i])
                    ymin.append(values[i+1])
                    xmax.append(values[i+2])
                    ymax.append(values[i+3])
                    label.append(values[i+4])
                    i+=5
                return np.array(xmin,dtype='float32'),np.array(xmax,dtype='float32'),np.array(ymin,dtype='float32'),np.array(ymax,dtype='float32'),np.array(label,dtype='float32')

            xmin,xmax,ymin,ymax,label=tf.py_function(get_data,[values[1:]],[tf.float32, tf.float32, tf.float32,tf.float32, tf.float32])
            image, bbox = get_random_data(tf.io.read_file(values[0]),xmin,xmax,ymin,ymax,label,input_shape, train=train)
            y0, y1, y2 = tf.py_function(preprocess_true_boxes, [bbox, input_shape, anchors, num_classes],
                                        [tf.float32, tf.float32, tf.float32])
            return (image, y0, y1, y2), 0

        if train:
            train_sum = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).shuffle(train_sum).map(parse, num_parallel_calls=AUTOTUNE).repeat().prefetch(batch_size).batch(batch_size)
        else:
            dataset = dataset.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).map(parse, num_parallel_calls=AUTOTUNE).repeat().prefetch(batch_size).batch(batch_size)
        #draw_image(dataset)
        return dataset

def auto_dataset(dataset_path:str,glob:str,batch_size,input_shape,anchors,num_classes,train: bool = True):
    files = tf.io.gfile.glob(os.path.join(dataset_path, glob))
    num = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
    if glob.endswith('.tfrecords'):
        return tfrecord_dataset(files, batch_size, input_shape, anchors, num_classes,train),num
    elif glob.endswith('.txt'):
        return text_dataset(files, batch_size, input_shape, anchors, num_classes,train),num