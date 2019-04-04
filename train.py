"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
import datetime
from yolo3.model import darknet_yolo_body, mobilenetv2_yolo_body, inception_yolo_body,densenet_yolo_body, yolo_loss
from typing import Tuple, List
from yolo3.data import auto_dataset
from yolo3.enum import OPT,BACKBONE
import os
from tensorflow.python import debug as tf_debug

gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_num = len(gpus.split(','))
if hasattr(tf,'enable_eager_execution'):
    tf.enable_eager_execution()

def _main():
    opt=None
    backbone = BACKBONE.MOBILENETV2
    log_dir = 'logs/'+str(backbone).split('.')[1].lower()+str(datetime.date.today())
    batch_size = 4
    train_dataset_path = '../pascal/VOCdevkit/train'
    val_dataset_path = '../pascal/VOCdevkit/val'
    train_dataset_glob='*2007*.tfrecords'
    val_dataset_glob='*2007*.tfrecords'
    model_config = {
        BACKBONE.MOBILENETV2: {
            "input_size": (320, 320),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        },
        BACKBONE.DARKNET53: {
            "input_size": (416, 416),
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
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt'
        }
    }

    class_names = get_classes(model_config[backbone]['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(model_config[backbone]['anchors_path'])
    input_shape = model_config[backbone]['input_size']  # multiple of 32, hw

    train_dataset,train_num=auto_dataset(train_dataset_path, train_dataset_glob, batch_size, input_shape, anchors, num_classes)
    val_dataset,val_num=auto_dataset(val_dataset_path, val_dataset_glob, batch_size, input_shape, anchors, num_classes,train=False)

    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    #with strategy.scope():
    if backbone == BACKBONE.MOBILENETV2:
        model = create_mobilenetv2_model(input_shape, anchors, num_classes, False, alpha=1.4,
                                         freeze_body=1, weights_path=model_config[backbone]['model_path'])
    elif backbone == BACKBONE.DARKNET53:
        model = create_darknet_model(input_shape, anchors, num_classes,
                                     freeze_body=1,
                                     weights_path=model_config[backbone]['model_path'])
    elif backbone == BACKBONE.INCEPTION_RESNET2:
        model = create_inception_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_config[backbone]['model_path'])
    elif backbone == BACKBONE.DENSENET:
        model = create_densenet_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_config[backbone]['model_path'])
    if opt==OPT.DEBUG:
        tf.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),'localhost:6064'))
    elif opt==OPT.XLA:
        config=tf.ConfigProto
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess=tf.Session(config=config)
        tf.keras.backend.set_session(sess)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir,write_grads=True,write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                    monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                    period=3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        #with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        model.fit(train_dataset,
                  epochs=10, initial_epoch=0,
                  steps_per_epoch=max(1, train_num // batch_size),
                  callbacks=[logging, checkpoint],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num // batch_size))
        model.save_weights(log_dir + str(backbone).split('.')[1].lower() + '_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
       # with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')
        model.fit(train_dataset,
                  epochs=20, initial_epoch=10, steps_per_epoch=max(1, train_num // batch_size),
                  callbacks=[logging,checkpoint, reduce_lr, early_stopping],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num // batch_size))
        model.save_weights(log_dir + str(backbone).split('.')[1].lower() + '_trained_weights_final.h5')

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

if __name__ == '__main__':
    _main()
