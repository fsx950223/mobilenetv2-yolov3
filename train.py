"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
import datetime
from yolo3.model import darknet_yolo_body, mobilenetv2_yolo_body, inception_yolo_body,densenet_yolo_body, yolo_loss,yolo_head
from typing import Tuple, List
from yolo3.data import Dataset
from yolo3.enum import OPT,BACKBONE
from yolo3.map import MAPCallback
from yolo3.tensorboard import log_scalar, log_images, log_histogram
import os
from tensorflow.python import debug as tf_debug

if hasattr(tf,'enable_eager_execution'):
    tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_num = len(gpus.split(','))
tf.keras.backend.set_learning_phase(1)
def _main():
    opt=OPT.DEBUG
    backbone = BACKBONE.MOBILENETV2
    log_dir = 'logs/'+str(backbone).split('.')[1].lower()+str(datetime.date.today())
    batch_size = 4
    train_dataset_path = '../pascal/VOCdevkit/train'
    val_dataset_path = '../pascal/VOCdevkit/val'
    test_dataset_path='../pascal/VOCdevkit/test'
    train_dataset_glob='*2007*.tfrecords'
    val_dataset_glob='*2007*.tfrecords'
    test_dataset_glob='*2007*.tfrecords'
    freeze_step=10
    train_step=20
    model_config = {
        BACKBONE.MOBILENETV2: {
            "input_size": [(416, 416),(224,224),(320,320),(512, 512),(608,608)],
            "model_path": '../download/ep049-loss10.452-val_loss9.514.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt',
            "learning_rate":[1e-4,1e-4],
            "alpha":1.4
        },
        BACKBONE.DARKNET53: {
            "input_size": (416, 416),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt',
            "learning_rate": [1e-3, 1e-4]
        },
        BACKBONE.INCEPTION_RESNET2: {
            "input_size": (608, 608),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt',
            "learning_rate": [1e-3, 1e-4]
        },
        BACKBONE.DENSENET: {
            "input_size": (416, 416),
            "model_path": '../download/trained_weights_final6.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt',
            "learning_rate": [1e-3, 1e-4]
        }
    }

    if opt == OPT.DEBUG:
        tf.get_logger().setLevel(tf.logging.DEBUG)
        tf.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(config=tf.ConfigProto(log_device_placement=True)), 'localhost:6064'))
        #tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    elif opt == OPT.XLA:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

    class_names = get_classes(model_config[backbone]['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(model_config[backbone]['anchors_path'])
    input_shape = model_config[backbone]['input_size']  # multiple of 32, hw
    model_path=model_config[backbone]['model_path']
    alpha=model_config[backbone]['alpha']
    lr = model_config[backbone]['learning_rate']

    train_dataset_callback=Dataset(os.path.join(train_dataset_path, train_dataset_glob), batch_size, input_shape, anchors,num_classes)
    train_dataset, train_num=train_dataset_callback.build()
    val_dataset,val_num=Dataset(os.path.join(val_dataset_path, val_dataset_glob),batch_size, input_shape, anchors,num_classes,train=False).build()
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync

    #with strategy.scope():
    #with tf.device('/cpu:0'):
    if backbone == BACKBONE.MOBILENETV2:
        model = create_mobilenetv2_model(anchors, num_classes, False, alpha=alpha,
                                         freeze_body=1, weights_path=model_path)
    elif backbone == BACKBONE.DARKNET53:
        model = create_darknet_model(input_shape, anchors, num_classes,
                                     freeze_body=1,
                                     weights_path=model_path)
    elif backbone == BACKBONE.INCEPTION_RESNET2:
        model = create_inception_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_path)
    elif backbone == BACKBONE.DENSENET:
        model = create_densenet_model(input_shape, anchors, num_classes, False,
                                       freeze_body=1,
                                       weights_path=model_path)

    def parse_fn(example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_image(features['image/encoded'],channels=3,dtype=tf.float32)
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
        label = tf.expand_dims(features['image/object/bbox/label'].values, 0)
        bbox = tf.concat([xmin,ymin,xmax,ymax,tf.cast(label, tf.float32)], 0)
        return image, bbox

    map_callback = MAPCallback(os.path.join(test_dataset_path, test_dataset_glob), input_shape, anchors, class_names,parse_fn, score=0,batch_size=batch_size)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir,write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir,'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                    period=1)
    cos_lr=tf.keras.callbacks.LearningRateScheduler(lambda epoch,_:tf.train.cosine_decay_restarts(lr[1],epoch-freeze_step,train_step//5)().numpy(),1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if gpu_num>=2:
        parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=gpu_num)
    else:
        parallel_model=model
    if True:
        #with strategy.scope():
        parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],epsilon=1e-8),
                  loss=[lambda y_true, y_pred: yolo_loss(y_pred, y_true, anchors,
                                                             print_loss=True)]*3)
        parallel_model.fit(train_dataset,
                  epochs=freeze_step, initial_epoch=0,
                  steps_per_epoch=max(1, train_num// batch_size),
                  callbacks=[logging,train_dataset_callback],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num// batch_size))
        model.save_weights(os.path.join(log_dir,str(backbone).split('.')[1].lower() + '_trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        #with strategy.scope():
        parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr[1],epsilon=1e-8),
                  loss=[lambda y_true, y_pred: yolo_loss(y_pred, y_true, anchors,
                                                             print_loss=True)]*3)  # recompile to apply the change
        print('Unfreeze all of the layers.')
        parallel_model.fit(train_dataset,
                  epochs=train_step+freeze_step, initial_epoch=freeze_step, steps_per_epoch=max(1, train_num // batch_size),
                  callbacks=[checkpoint,cos_lr,map_callback,train_dataset_callback,early_stopping],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num // batch_size))
        model.save_weights(os.path.join(log_dir,str(backbone).split('.')[1].lower() + '_trained_weights_final.h5'))

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
    return model_body

def create_mobilenetv2_model(anchors, num_classes, load_pretrained: bool = True,
                             freeze_body: int = 2, alpha: float = 1.0,
                             weights_path: str = 'model_data/tiny_yolo_weights.h5'):
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(None, None, 3))
    model_body = mobilenetv2_yolo_body(x_data, num_anchors // 3, num_classes, alpha)
    print('Create MobilenetV2-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        num = (155, len(model_body.layers) - 3)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    return model_body

def create_inception_model(input_shape, anchors, num_classes, load_pretrained: bool = True,
                           freeze_body: int = 2,
                           weights_path: str = 'model_data/tiny_yolo_weights.h5'):
    h, w = input_shape
    num_anchors = len(anchors)
    x_data = tf.keras.layers.Input(shape=(h, w, 3))
    h = h - 64
    w = w - 64
    y_data = [tf.keras.layers.Input(shape=(h // [32, 16, 8][l], w // [32, 16, 8][l], \
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
    model_body = densenet_yolo_body(x_data, num_anchors // 3, num_classes)
    print('Create Densenet-YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
    if freeze_body in [1, 2]:
        num = (707, len(model_body.layers) - 3)[freeze_body - 1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    y1 = tf.keras.layers.Lambda(
        lambda output: [yolo_head(output, anchors[anchor_mask[0]], num_classes, input_shape, True)[1]], name='y1')(
        model_body.output[0])
    y2 = tf.keras.layers.Lambda(
        lambda output: [yolo_head(output, anchors[anchor_mask[1]], num_classes, input_shape, True)[1]], name='y2')(
        model_body.output[1])
    y3 = tf.keras.layers.Lambda(
        lambda output: [yolo_head(output, anchors[anchor_mask[2]], num_classes, input_shape, True)[1]], name='y3')(
        model_body.output[2])
    model = tf.keras.models.Model(model_body.input, [y1, y2, y3])
    return model

if __name__ == '__main__':
    _main()
