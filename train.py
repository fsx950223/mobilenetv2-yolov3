"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import datetime
import zipfile
from yolo3.model import darknet_yolo_body, YoloLoss, mobilenetv2_yolo_body, efficientnet_yolo_body
from yolo3.data import Dataset
from yolo3.enum import OPT, BACKBONE, DATASET_MODE
from yolo3.map import MAPCallback
from yolo3.utils import get_anchors, get_classes
import os
import numpy as np
from tensorflow.python import debug as tf_debug


AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.keras.backend.set_learning_phase(1)


def train(FLAGS):
    prune = False
    opt = FLAGS.opt or None
    backbone = FLAGS.backbone or BACKBONE.MOBILENETV2
    log_dir = os.path.join('logs',str(backbone).split('.')[1].lower()+str(datetime.date.today()))
    if tf.io.gfile.exists(log_dir) is not True:
        tf.io.gfile.mkdir(log_dir)
    batch_size = FLAGS.batch_size or 8
    train_dataset_glob=FLAGS.train_dataset or '../pascal/VOCdevkit/train/*2007*.tfrecords'
    val_dataset_glob=FLAGS.val_dataset or '../pascal/VOCdevkit/val/*2007*.tfrecords'
    test_dataset_glob=FLAGS.test_dataset or '../pascal/VOCdevkit/test/*2007*.tfrecords'
    freeze_step = FLAGS.epochs[0] or 10
    train_step = FLAGS.epochs[1] or 10
    model_config = {
        BACKBONE.MOBILENETV2: {
            # "input_size": [(416, 416), (224, 224), (320, 320), (512, 512),
            #                (608, 608)],
            "input_size":(224,224),
            "model_path": None,
            "anchors_path":
            'model_data/yolo_anchors.txt',
            "classes_path":
            'model_data/voc_classes.txt',
            "learning_rate": [1e-4, 1e-4],
            "alpha":
            1.4
        },
        BACKBONE.DARKNET53: {
            "input_size": [(416, 416), (224, 224), (320, 320), (512, 512),
                           (608, 608)],
            "model_path":
            None,
            "anchors_path":
            'model_data/yolo_anchors.txt',
            "classes_path":
            'model_data/voc_classes.txt',
            "learning_rate": [1e-4, 1e-4]
        },
        BACKBONE.EFFICIENTNET: {
            "input_size": (380, 380),
            "model_path": None,
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/voc_classes.txt',
            "learning_rate": [1e-3, 1e-4]
        }
    }

    if opt == OPT.DEBUG:
        tf.get_logger().setLevel(tf.logging.DEBUG)
        tf.keras.backend.set_session(
            tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    elif opt == OPT.XLA:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

    class_names = get_classes(model_config[backbone]['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(model_config[backbone]['anchors_path'])
    input_shape = model_config[backbone]['input_size']  # multiple of 32, hw
    model_path = model_config[backbone]['model_path']
    lr = model_config[backbone]['learning_rate']

    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset_callback = Dataset(train_dataset_glob,
                                     batch_size,
                                     anchors,
                                     num_classes,
                                     input_shape)
    train_dataset, train_num = train_dataset_callback.build()
    val_dataset_builder = Dataset(val_dataset_glob,
                                  batch_size,
                                  anchors,
                                  num_classes,
                                  input_shape,
                                  mode=DATASET_MODE.VALIDATE)
    val_dataset, val_num = val_dataset_builder.build()
    map_callback = MAPCallback(test_dataset_glob,
                               input_shape,
                               anchors,
                               class_names)
    logging = tf.keras.callbacks.TensorBoard(write_graph=False,log_dir=log_dir, write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    period=3)
    if tf.version.VERSION.startswith('1.'):
        cos_lr = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, _: tf.train.cosine_decay(lr[1], epoch - freeze_step,
                                                   train_step)().numpy(), 1)
    else:
        cos_lr = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, _: tf.keras.experimental.CosineDecay(
                lr[1], train_step)(epoch - freeze_step).numpy(), 1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=(freeze_step + train_step) // 10,
        verbose=1)
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if tf.version.VERSION.startswith('1.'):
        loss = [
            lambda y_true, yolo_output: YoloLoss(
                y_true, yolo_output, 0, anchors, print_loss=True), lambda
            y_true, yolo_output: YoloLoss(
                y_true, yolo_output, 1, anchors, print_loss=True), lambda
            y_true, yolo_output: YoloLoss(
                y_true, yolo_output, 2, anchors, print_loss=True)
        ]
    else:
        loss = [YoloLoss(idx, anchors, print_loss=True) for idx in range(3)]

    with strategy.scope():
        factory = ModelFactory(weights_path=model_path)
        if backbone == BACKBONE.MOBILENETV2:
            model = factory.build(mobilenetv2_yolo_body,
                                  155,
                                  len(anchors) // 3,
                                  num_classes,
                                  alpha=model_config[backbone]['alpha'])
        elif backbone == BACKBONE.DARKNET53:
            model = factory.build(darknet_yolo_body, 185,
                                  len(anchors) // 3, num_classes)
        elif backbone == BACKBONE.EFFICIENTNET:
            override_params = {}
            override_params['batch_norm_momentum'] = 0.9
            override_params['batch_norm_epsilon'] = 1e-3
            override_params['num_classes']=num_classes
            override_params['drop_connect_rate'] = 0.2
            override_params['data_format']='channels_first'
            model=factory.build(efficientnet_yolo_body,499,'efficientnet-b4',len(anchors) // 3,batch_norm_momentum=0.9,batch_norm_epsilon=1e-3,num_classes=num_classes,drop_connect_rate=0.2,data_format="channels_first")

    if prune:
        from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
        end_step = np.ceil(1.0 * train_num / batch_size).astype(
            np.int32) * train_step
        new_pruning_params = {
            'pruning_schedule':
            sparsity.PolynomialDecay(initial_sparsity=0.5,
                                     final_sparsity=0.9,
                                     begin_step=0,
                                     end_step=end_step,
                                     frequency=1000)
        }
        pruned_model = sparsity.prune_low_magnitude(model,
                                                    **new_pruning_params)
        pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],
                                                                epsilon=1e-8),
                             loss=loss)
        pruned_model.fit(train_dataset,
                         epochs=train_step,
                         initial_epoch=0,
                         steps_per_epoch=max(1, train_num // batch_size),
                         callbacks=[
                             checkpoint, cos_lr, logging, map_callback,
                             train_dataset_callback, early_stopping
                         ],
                         validation_data=val_dataset,
                         validation_steps=max(1, val_num // batch_size))
        model = sparsity.strip_pruning(pruned_model)
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_pruned.h5'))
        with zipfile.ZipFile(os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_pruned.h5.zip'),
                             'w',
                             compression=zipfile.ZIP_DEFLATED) as f:
            f.write(
                os.path.join(
                    log_dir,
                    str(backbone).split('.')[1].lower() +
                    '_trained_weights_pruned.h5'))
        return

    if True:
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],epsilon=1e-8),
                                loss=loss)
        model.fit(
            train_dataset,
            epochs=freeze_step,
            initial_epoch=0,
            steps_per_epoch=max(1, train_num // batch_size),
            callbacks=[logging, checkpoint, train_dataset_callback],
            validation_data=val_dataset,
            validation_steps=max(1, val_num // batch_size))
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_stage_1.h5'))
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[1],epsilon=1e-8),
                               loss=loss)  # recompile to apply the change
        print('Unfreeze all of the layers.')
        model.fit(train_dataset,
                           epochs=train_step + freeze_step,
                           initial_epoch=freeze_step,
                           steps_per_epoch=max(1, train_num // batch_size),
                           callbacks=[
                               checkpoint, cos_lr, logging, map_callback,
                               train_dataset_callback, early_stopping
                           ],
                           validation_data=val_dataset,
                           validation_steps=max(1, val_num // batch_size))
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_final.h5'))

    # Further training if needed.


class ModelFactory(object):

    def __init__(self,
                 input=tf.keras.layers.Input(shape=(None, None, 3)),
                 weights_path=None):
        self.input = input
        self.weights_path = weights_path

    def build(self, model_builder, freeze_layers=None, *args, **kwargs):
        model_body = model_builder(self.input, *args, **kwargs)
        if self.weights_path is not None:
            model_body.load_weights(self.weights_path, by_name=True)
            print('Load weights {}.'.format(self.weights_path))
        # Freeze the darknet body or freeze all but 2 output layers.
        freeze_layers = freeze_layers or len(model_body.layers) - 3
        for i in range(freeze_layers):
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(
            freeze_layers, len(model_body.layers)))
        return model_body
