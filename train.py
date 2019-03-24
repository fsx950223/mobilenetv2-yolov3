"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf

from yolo3.model import preprocess_true_boxes, darknet_yolo_body, mobilenetv2_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from typing import Tuple, List
from multiprocessing import cpu_count
import os
from functools import reduce
gpus="0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
tf.enable_eager_execution()
def _main():
    log_dir = 'logs/000/'
    classes_path = '../pascal/VOCdevkit/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (224, 224)  # multiple of 32, hw
    batch_size = 4
    train_dataset_path='../pascal/VOCdevkit/train'
    val_dataset_path = '../pascal/VOCdevkit/val'

    is_tiny_version = True  # default setting
    files = tf.gfile.Glob(os.path.join(train_dataset_path, '*VOC2007*.tfrecords'))
    sum = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
    val_files = tf.gfile.Glob(os.path.join(val_dataset_path, '*VOC2007*.tfrecords'))
    val_sum = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), val_files))
    if is_tiny_version:
        model = create_mobilenetv2_model(input_shape, anchors, num_classes, False, alpha=1.4,
                                  freeze_body=1, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_darknet_model(train_dataset_path,batch_size, input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path='model_data/darknet53_weights.h5')  # make sure you know what you freeze
    is_multi_gpu=len(gpus.split(','))>1
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                    monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                    period=3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if is_multi_gpu:
        strategy = tf.distribute.MirroredStrategy()

    if True:
        model.compile(optimizer=tf.train.AdamOptimizer(1e-3),loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred},
                      distribute=strategy if is_multi_gpu else None)

        model.fit(data_generator(files,batch_size, input_shape, anchors, num_classes),
                    epochs=5, initial_epoch=0,
                    steps_per_epoch=max(1, sum // batch_size),
                    callbacks=[logging, checkpoint],
                    validation_data=data_generator(val_files, batch_size, input_shape, anchors,num_classes,train=False),
                    validation_steps=max(1, val_sum // batch_size))
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=tf.train.AdamOptimizer(1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred},distribute=strategy if is_multi_gpu else None)  # recompile to apply the change
        print('Unfreeze all of the layers.')

        model.fit(data_generator(files,batch_size, input_shape, anchors, num_classes),
                    epochs=10, initial_epoch=5, steps_per_epoch=max(1, sum // batch_size),
                    callbacks=[checkpoint, reduce_lr, early_stopping],
                    validation_data=data_generator(val_files, batch_size, input_shape, anchors, num_classes,train=False),
                    validation_steps=max(1, val_sum // batch_size))
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path: str) -> List[str]:
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path: str) -> List[List[float]]:
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_darknet_model(input_shape: Tuple[int, int], anchors: List[List[float]], num_classes: int,
                 load_pretrained: bool = True,
                 freeze_body: int = 2,
                 weights_path: str = 'model_data/yolo_weights.h5') -> tf.keras.models.Model:
    """create the training model"""
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(None, None, 3))
    y_data = [tf.keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = darknet_yolo_body(x_data, num_anchors // 3, num_classes)
    print('Create Darknet-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        # Freeze the darknet body or freeze all but 2 output layers.
        num = (155, len(model_body.layers) - 2)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5})(
        [*model_body.output, *y_data])
    model = tf.keras.models.Model([model_body.input, *y_data], model_loss)
    return model

def create_mobilenetv2_model(input_shape, anchors, num_classes, load_pretrained: bool = True,
                      freeze_body: int = 2, alpha: float = 1.0,
                      weights_path: str = 'model_data/tiny_yolo_weights.h5'):
    tf.keras.backend.clear_session()
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(None, None, 3))
    y_data = [tf.keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = mobilenetv2_yolo_body(x_data,num_anchors // 3, num_classes, alpha)
    print('Create MobilenetV2-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        # Freeze the darknet body or freeze all but 2 output layers.
        num = (155, len(model_body.layers) - 2)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5})(
        [*model_body.output, *y_data])
    model = tf.keras.models.Model([model_body.input, *y_data], model_loss)
    return model


def data_generator(files: List[str], batch_size: int, input_shape: Tuple[int, int],
                   anchors: List[float],
                   num_classes: int,
                   train:bool=True):
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        """data generator for fit_generator"""
        def parse(example_proto):
            feature_description = {
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                'image/object/bbox/label': tf.VarLenFeature(tf.int64)
            }
            features = tf.parse_single_example(example_proto, feature_description)
            image, bbox = get_random_data(features, input_shape,train=train)
            y0, y1, y2 = tf.py_function(preprocess_true_boxes, [bbox, input_shape, anchors, num_classes],
                                        [tf.float32, tf.float32, tf.float32])
            return (image, y0, y1, y2),0
        if train:
            dataset = dataset.interleave(lambda x:tf.data.TFRecordDataset(x).map(parse,num_parallel_calls=cpu_count()),cycle_length=len(files)).shuffle(300).prefetch(batch_size).repeat().batch(batch_size)
        else:
            dataset = dataset.interleave(lambda x:tf.data.TFRecordDataset(x).map(parse,num_parallel_calls=cpu_count()),cycle_length=len(files)).repeat().batch(batch_size).prefetch(
                batch_size)
        return dataset

if __name__ == '__main__':
    _main()
