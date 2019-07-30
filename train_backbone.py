import tensorflow as tf
from yolo3.data import Dataset
from yolo3.override import mobilenet_v2
from yolo3.darknet import darknet_body
from yolo3.efficientnet import EfficientNetB4
from yolo3.utils import get_classes, ModelFactory
from yolo3.enum import BACKBONE
import os
import datetime


class BackboneDataset(Dataset):
    """Backbone's Dataset extends Dataset,only support txt files now.
    """
    def parse_tfrecord(self, example_proto):
        pass

    def parse_text(self, line):
        values = tf.strings.split([line], ' ').values
        image = tf.image.decode_image(tf.io.read_file(values[0]),
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize(image, self.input_shape)
        label = tf.strings.to_number(values[1], tf.int64)
        return image, label


def mobilenetv2(inputs, alpha, classes):
    """MobilenetV2 wrapper function
    
    Arguments:
        inputs {np.array} -- [train images]
        alpha {float} -- [controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.]
        classes {int} -- [classes total number]
    
    Returns:
        [tf.keras.Model] -- [mobilenetv2 model]
    """
    return mobilenet_v2(default_batchnorm_momentum=0.9,
                        alpha=alpha,
                        input_tensor=inputs,
                        classes=classes)


def EfficientNet(inputs, classes, input_shape):
    return EfficientNetB4(classes=classes,
                          input_shape=input_shape,
                          input_tensor=inputs)


def train(FLAGS):
    batch_size = FLAGS['batch_size']
    use_tpu = FLAGS['use_tpu']
    class_names = get_classes(FLAGS['classes_path'])
    epochs = FLAGS['epochs'][0]
    input_size = FLAGS['input_size']
    model_path = FLAGS['model']
    backbone = FLAGS['backbone']
    train_dataset_glob = FLAGS['train_dataset']
    val_dataset_glob = FLAGS['val_dataset']
    log_dir = FLAGS['log_directory'] or os.path.join(
        'logs',
        str(backbone).split('.')[1].lower() + str(datetime.date.today()))
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        factory = ModelFactory(weights_path=model_path)
        if backbone == BACKBONE.MOBILENETV2:
            model = factory.build(mobilenetv2,
                                  0,
                                  alpha=1.4,
                                  classes=len(class_names))
        elif backbone == BACKBONE.DARKNET53:
            model = factory.build(darknet_body, 0, classes=len(class_names))
        elif backbone == BACKBONE.EFFICIENTNET:
            model = factory.build(EfficientNet,
                                  0,
                                  classes=len(class_names),
                                  input_shape=(*input_size, 3))
        model.compile(tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    if use_tpu:
        tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
        tpu_strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=tpu_strategy)

    train_dataset, train_num = BackboneDataset(train_dataset_glob,
                                               batch_size,
                                               num_classes=len(class_names),
                                               input_shapes=input_size).build()
    val_dataset, val_num = BackboneDataset(val_dataset_glob,
                                           batch_size,
                                           num_classes=len(class_names),
                                           input_shapes=input_size).build()

    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.train.cosine_decay(1e-3, epoch, epochs)().numpy(),
        1)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=3)
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=max(1, train_num // batch_size),
              validation_data=val_dataset,
              validation_steps=max(1, val_num // batch_size),
              callbacks=[cos_lr, logging, checkpoint])
    model.save_weights(
        os.path.join(
            log_dir,
            str(backbone).split('.')[1].lower() + '_trained_weights.h5'))
