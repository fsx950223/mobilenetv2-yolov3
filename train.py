"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import datetime
from yolo3.model import darknet_yolo_body, YoloLoss, mobilenetv2_yolo_body, efficientnet_yolo_body
from yolo3.data import Dataset
from yolo3.enums import BACKBONE, DATASET_MODE
from yolo3.map import MAPCallback
from yolo3.utils import get_anchors, get_classes, ModelFactory
import os
import numpy as np
import neural_structured_learning as nsl

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.keras.backend.set_learning_phase(1)


def train(FLAGS):
    """Train yolov3 with different backbone
    """
    prune = FLAGS['prune']
    opt = FLAGS['opt']
    backbone = FLAGS['backbone']
    log_dir = os.path.join(
        'logs',
        str(backbone).split('.')[1].lower() + '_' + str(datetime.date.today()))

    batch_size = FLAGS['batch_size']
    train_dataset_glob = FLAGS['train_dataset']
    val_dataset_glob = FLAGS['val_dataset']
    test_dataset_glob = FLAGS['test_dataset']
    freeze = FLAGS['freeze']
    epochs = FLAGS['epochs'][0] if freeze else FLAGS['epochs'][1]

    class_names = get_classes(FLAGS['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(FLAGS['anchors_path'])
    input_shape = FLAGS['input_size']  # multiple of 32, hw
    model_path = FLAGS['model']
    if model_path and model_path.endswith(
            '.h5') is not True:
        model_path = tf.train.latest_checkpoint(model_path)
    lr = FLAGS['learning_rate']
    tpu_address=FLAGS['tpu_address']
    if tpu_address is not None:
        cluster_resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_host(cluster_resolver.master())
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy=tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy(devices=FLAGS['gpus'])
    batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset_builder = Dataset(train_dataset_glob, batch_size, anchors,
                                     num_classes, input_shape)
    train_dataset, train_num = train_dataset_builder.build(epochs)
    val_dataset_builder = Dataset(val_dataset_glob,
                                  batch_size,
                                  anchors,
                                  num_classes,
                                  input_shape,
                                  mode=DATASET_MODE.VALIDATE)
    val_dataset, val_num = val_dataset_builder.build(epochs)
    map_callback = MAPCallback(test_dataset_glob, input_shape, anchors,
                               class_names)
    tensorboard = tf.keras.callbacks.TensorBoard(write_graph=False,
                                             log_dir=log_dir,
                                             write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    period=3)
    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[1], epochs)(
            epoch).numpy(),1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=epochs // 5,
        verbose=1)

    loss = [YoloLoss(idx, anchors, print_loss=False) for idx in range(len(anchors) // 3)]

    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=0.2, adv_step_size=0.2, adv_grad_norm='infinity')
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    with strategy.scope():
        factory = ModelFactory(tf.keras.layers.Input(shape=(*input_shape, 3)),
                               weights_path=model_path)
        if backbone == BACKBONE.MOBILENETV2:
            model = factory.build(mobilenetv2_yolo_body,
                                  155,
                                  len(anchors) // 3,
                                  num_classes,
                                  alpha=FLAGS['alpha'])
        elif backbone == BACKBONE.DARKNET53:
            model = factory.build(darknet_yolo_body, 185,
                                  len(anchors) // 3, num_classes)
        elif backbone == BACKBONE.EFFICIENTNET:
            model = factory.build(efficientnet_yolo_body,
                                  499,
                                  FLAGS['model_name'],
                                  len(anchors) // 3,
                                  batch_norm_momentum=0.9,
                                  batch_norm_epsilon=1e-3,
                                  num_classes=num_classes,
                                  drop_connect_rate=0.2,
                                  data_format="channels_first")

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if freeze is True:
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],
                                                             epsilon=1e-8),
                          loss=loss)
        model.fit(epochs, [checkpoint, tensorboard, tf.keras.callbacks.LearningRateScheduler(
        (lambda _, lr:lr),1)], train_dataset, val_dataset)
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_stage_1.h5'))
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[1],
                                                             epsilon=1e-8),
                          loss=loss)  # recompile to apply the change
        print('Unfreeze all of the layers.')
        model.fit(epochs, [checkpoint, cos_lr, tensorboard, early_stopping], train_dataset,
                      val_dataset,use_adv=False)
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_final.h5'))

    # Further training if needed.
