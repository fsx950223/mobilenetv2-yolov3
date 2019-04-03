"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
import datetime
from yolo3.model import preprocess_true_boxes, darknet_yolo_body, mobilenetv2_yolo_body, inception_yolo_body,densenet_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from typing import Tuple, List
from multiprocessing import cpu_count
import os
from functools import reduce
from tensorflow.python import debug as tf_debug
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_num = len(gpus.split(','))
if hasattr(tf,'enable_eager_execution'):
    tf.enable_eager_execution()

def _main():
    opt=None
    backbone = "mobilenetv2"
    log_dir = 'logs/'+backbone+str(datetime.date.today())
    batch_size = 4
    train_dataset_path = '../pascal/VOCdevkit/train'
    val_dataset_path = '../pascal/VOCdevkit/val'
    model_config = {
        "mobilenetv2": {
            "input_size": (224, 224),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        },
        "darknet53": {
            "input_size": (416, 416),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        },
        "inception": {
            "input_size": (608, 608),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        },
        "densenet": {
            "input_size": (416, 416),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        }
    }

    class_names = get_classes(model_config[backbone]['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(model_config[backbone]['anchors_path'])
    input_shape = model_config[backbone]['input_size']  # multiple of 32, hw


    files = tf.io.gfile.glob(os.path.join(train_dataset_path, '*2007*.tfrecords'))
    train_sum = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
    val_files = tf.io.gfile.glob(os.path.join(val_dataset_path, '*2007*.tfrecords'))
    val_sum = reduce(lambda x, y: x + y,
                     map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), val_files))
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    #with strategy.scope():
    if backbone == "mobilenetv2":
        model = create_mobilenetv2_model(input_shape, anchors, num_classes, False, alpha=1.4,
                                         freeze_body=1, weights_path=model_config[backbone]['model_path'])
    elif backbone == "darknet53":
        model = create_darknet_model(input_shape, anchors, num_classes,
                                     freeze_body=1,
                                     weights_path=model_config[backbone]['model_path'])
    elif backbone == "inception":
        model = create_inception_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_config[backbone]['model_path'])
    elif backbone == "densenet":
        model = create_densenet_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_config[backbone]['model_path'])
    if opt=="debug":
        tf.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),'localhost:6064'))
    elif opt=="xla":
        config=tf.ConfigProto
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess=tf.Session(config=config)
        tf.keras.backend.set_session(sess)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir,write_grads=True,write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                    monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                    period=3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.

    if True:
        #with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        model.fit(data_generator(files, batch_size, input_shape, anchors, num_classes),
                  epochs=10, initial_epoch=0,
                  steps_per_epoch=max(1, train_sum // batch_size),
                  callbacks=[logging, checkpoint],
                  validation_data=data_generator(val_files, batch_size, input_shape, anchors, num_classes, train=False),
                  validation_steps=max(1, val_sum // batch_size))
        model.save_weights(log_dir + backbone + '_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
       # with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')
        model.fit(data_generator(files, batch_size, input_shape, anchors, num_classes),
                  epochs=20, initial_epoch=10, steps_per_epoch=max(1, train_sum // batch_size),
                  callbacks=[logging,checkpoint, reduce_lr, early_stopping],
                  validation_data=data_generator(val_files, batch_size, input_shape, anchors, num_classes, train=False),
                  validation_steps=max(1, val_sum // batch_size))
        model.save_weights(log_dir + backbone + '_trained_weights_final.h5')

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
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(h, w, 3))
    y_data = [tf.keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = darknet_yolo_body(x_data, num_anchors // 3, num_classes)
    print('Create Darknet-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        # Freeze the darknet body or freeze all but 2 output layers.
        num = (185, len(model_body.layers) - 3)[freeze_body - 1]
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
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(h, w, 3))
    y_data = [tf.keras.layers.Input(shape=(h // [32, 16, 8][l], w // [32, 16, 8][l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = mobilenetv2_yolo_body(x_data, num_anchors // 3, num_classes, alpha)
    print('Create MobilenetV2-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        num = (155, len(model_body.layers) - 3)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5,'print_loss':True})(
        [*model_body.output, *y_data])
    model = tf.keras.models.Model([model_body.input, *y_data], model_loss)
    return model


def create_inception_model(input_shape, anchors, num_classes, load_pretrained: bool = True,
                           freeze_body: int = 2,
                           weights_path: str = 'model_data/tiny_yolo_weights.h5'):
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(h, w, 3))
    h = h - 64
    w = w - 64
    y_data = [tf.keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = inception_yolo_body(x_data, num_anchors // 3, num_classes)
    print('Create Inception-Res2-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        num = (780, len(model_body.layers) - 3)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5})(
        [*model_body.output, *y_data])
    model = tf.keras.models.Model([model_body.input, *y_data], model_loss)
    return model

def create_densenet_model(input_shape, anchors, num_classes, load_pretrained: bool = True,
                           freeze_body: int = 2,
                           weights_path: str = 'model_data/tiny_yolo_weights.h5'):
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(h, w, 3))
    y_data = [tf.keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = densenet_yolo_body(x_data, num_anchors // 3, num_classes)
    print('Create Densenet-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        num = (707, len(model_body.layers) - 3)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5})(
        [*model_body.output, *y_data])
    model = tf.keras.models.Model([model_body.input, *y_data], model_loss)
    return model

def draw_image(dataset):
    file_writer_cm = tf.summary.create_file_writer('logs/cm')
    iter=dataset.__iter__()
    images,_,_,_=iter.get_next()[0]
    images = np.reshape(images[0:25], (-1, 28, 28, 1))
    with file_writer_cm.as_default():
        tf.summary.image('input_image', images,max_outputs=25, step=0)

def data_generator(files: List[str], batch_size: int, input_shape: Tuple[int, int],
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
            image, bbox = get_random_data(features, input_shape, train=train)

            y0, y1, y2 = tf.py_function(preprocess_true_boxes, [bbox, input_shape, anchors, num_classes],
                                        [tf.float32, tf.float32, tf.float32])
            return (image, y0, y1, y2), 0

        if train:
            train_sum = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x).map(parse, num_parallel_calls=cpu_count()),
                cycle_length=len(files),num_parallel_calls=min(cpu_count(),len(files))).shuffle(300).repeat().prefetch(batch_size).batch(batch_size)
        else:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x).map(parse, num_parallel_calls=cpu_count()),
                cycle_length=len(files),num_parallel_calls=min(cpu_count(),len(files))).repeat().prefetch(batch_size).batch(batch_size)
        #draw_image(dataset)
        return dataset


if __name__ == '__main__':
    _main()
