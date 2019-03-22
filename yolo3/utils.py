"""Miscellaneous utility functions."""

from functools import reduce
from PIL import Image
import numpy as np
import tensorflow as tf


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(features, input_shape, hue=.1, sat=.1, val=.1, max_boxes=20,min_jpeg_quality=80,max_jpeg_quality=100, train:bool=True):
    '''random preprocessing for real-time data augmentation'''
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    iw, ih = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
    h, w = tf.cast(input_shape[0], tf.float32), tf.cast(input_shape[1], tf.float32)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    label = tf.expand_dims(features['image/object/bbox/label'].values, 0)

    nh = ih * tf.minimum(w / iw, h / ih)
    nw = iw * tf.minimum(w / iw, h / ih)
    # 将图片按照固定长宽比进行padding缩放
    dx = (w - nw) / 2
    dy = (h - nh) / 2
    image = tf.image.resize(image, [tf.cast(nh, tf.int32), tf.cast(nw, tf.int32)])
    new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                             tf.cast(h, tf.int32), tf.cast(w, tf.int32))
    # place image
    image_ones = tf.ones_like(image)
    image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                     tf.cast(h, tf.int32), tf.cast(w, tf.int32))
    image_color_padded = (1 - image_ones_padded) * 128
    image = tf.image.convert_image_dtype(image_color_padded, tf.float32) + new_image

    # flip image or not
    xmin = xmin * nw / iw + dx
    xmax = xmax * nw / iw + dx
    ymin = ymin * nh / ih + dy
    ymax = ymax * nh / ih + dy

    if train:
        image, xmin, xmax=tf.cond(tf.less(tf.random.uniform([]), 0.5),lambda: (tf.image.flip_left_right(image),w-xmax,w-xmin),lambda :(image,xmin,xmax))
        image = tf.image.random_hue(image, hue)
        image = tf.image.random_saturation(image, 1 - sat, 1 + sat)
        image = tf.image.random_brightness(image, val)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality)

    bbox = tf.concat([xmin, ymin, xmax, ymax, tf.cast(label, tf.float32)], 0)
    bbox = tf.transpose(bbox, [1, 0])

    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=input_shape[0] - 1)
    bbox = tf.cond(tf.greater(tf.shape(bbox)[0], max_boxes), lambda: bbox[:max_boxes], lambda: bbox)
    return image, bbox
