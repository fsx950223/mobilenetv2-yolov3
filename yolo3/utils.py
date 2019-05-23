"""Miscellaneous utility functions."""

from functools import reduce
import cv2
import tensorflow as tf
import numpy as np


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    if len(image.shape) == 4:
        iw, ih = tf.cast(tf.shape(image)[2],
                         tf.int32), tf.cast(tf.shape(image)[1], tf.int32)
    elif len(image.shape) == 3:
        iw, ih = tf.cast(tf.shape(image)[1],
                         tf.int32), tf.cast(tf.shape(image)[0], tf.int32)
    w, h = tf.cast(size[1], tf.int32), tf.cast(size[0], tf.int32)
    nh = tf.cast(tf.cast(ih, tf.float64) * tf.minimum(w / iw, h / ih), tf.int32)
    nw = tf.cast(tf.cast(iw, tf.float64) * tf.minimum(w / iw, h / ih), tf.int32)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    resized_image = tf.image.resize(image, [nh, nw])
    new_image = tf.image.pad_to_bounding_box(resized_image, dy, dx, h, w)
    image_color_padded = tf.cast(tf.equal(new_image, 0),
                                 tf.float32) * (128 / 255)
    return image_color_padded + new_image, tf.shape(resized_image)


def random_gamma(image, min, max):
    val = tf.random.uniform([], min, max)
    return tf.image.adjust_gamma(image, val)


def random_blur(image):
    gaussian_blur = lambda image: cv2.GaussianBlur(image.numpy(), (5, 5), 0)
    h, w = image.shape.as_list()[:2]
    image = tf.py_function(gaussian_blur, [image], tf.float32)
    image.set_shape([h, w, 3])
    return image


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def bind(instance, func, as_name=None):
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_random_data(image,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    label,
                    input_shape,
                    min_scale=0.25,
                    max_scale=2,
                    jitter=0.3,
                    min_gamma=0.8,
                    max_gamma=2,
                    blur=False,
                    flip=True,
                    hue=.5,
                    sat=.5,
                    val=0.,
                    cont=.1,
                    noise=0,
                    max_boxes=20,
                    min_jpeg_quality=80,
                    max_jpeg_quality=100,
                    train: bool = True):
    '''random preprocessing for real-time data augmentation'''

    iw, ih = tf.cast(tf.shape(image)[1],
                     tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
    w, h = tf.cast(input_shape[1], tf.float32), tf.cast(input_shape[0],
                                                        tf.float32)
    xmax = tf.expand_dims(xmax, 0)
    xmin = tf.expand_dims(xmin, 0)
    ymax = tf.expand_dims(ymax, 0)
    ymin = tf.expand_dims(ymin, 0)
    label = tf.expand_dims(label, 0)
    if train:
        new_ar = (w / h) * (tf.random.uniform([], 1 - jitter, 1 + jitter) /
                            tf.random.uniform([], 1 - jitter, 1 + jitter))
        scale = tf.random.uniform([], min_scale, max_scale)
        ratio = tf.cond(tf.less(
            new_ar, 1), lambda: scale * new_ar, lambda: scale / new_ar)
        ratio = tf.maximum(ratio, 1)
        nw, nh = tf.cond(tf.less(
            new_ar,
            1), lambda: (ratio * h, scale * h), lambda: (scale * w, ratio * w))
        dx = tf.random.uniform([], 0, w - nw)
        dy = tf.random.uniform([], 0, h - nh)
        image = tf.image.resize(image,
                                [tf.cast(nh, tf.int32),
                                 tf.cast(nw, tf.int32)])

        def crop_and_pad(image, dx, dy):
            dy = tf.cast(tf.math.maximum(-dy, 0), tf.int32)
            dx = tf.cast(tf.math.maximum(-dx, 0), tf.int32)
            image = tf.image.crop_to_bounding_box(
                image, dy, dx,
                tf.math.minimum(tf.cast(h, tf.int32), tf.cast(nh, tf.int32)),
                tf.math.minimum(tf.cast(w, tf.int32), tf.cast(nw, tf.int32)))
            image = tf.image.pad_to_bounding_box(image, 0, 0,
                                                 tf.cast(h, tf.int32),
                                                 tf.cast(w, tf.int32))
            return image

        new_image = tf.cond(
            tf.greater(scale,
                       1), lambda: crop_and_pad(image, dx, dy), lambda: tf.image
            .pad_to_bounding_box(image, tf.cast(tf.math.maximum(
                dy, 0), tf.int32), tf.cast(tf.math.maximum(dx, 0), tf.int32),
                                 tf.cast(h, tf.int32), tf.cast(w, tf.int32)))
        image_color_padded = tf.cast(tf.equal(new_image, 0),
                                     tf.float32) * (128 / 255)
        image = image_color_padded + new_image

        xmin = xmin * nw / iw + dx
        xmax = xmax * nw / iw + dx
        ymin = ymin * nh / ih + dy
        ymax = ymax * nh / ih + dy
        if flip:
            image, xmin, xmax = tf.cond(
                tf.less(
                    tf.random.uniform([]),
                    0.5), lambda: (tf.image.flip_left_right(image), w - xmax, w
                                   - xmin), lambda: (image, xmin, xmax))
        if hue > 0:
            image = tf.image.random_hue(image, hue)
        if sat > 1:
            image = tf.image.random_saturation(image, 1 - sat, 1 + sat)
        if val > 0:
            image = tf.image.random_brightness(image, val)
        if min_gamma < max_gamma:
            image = random_gamma(image, min_gamma, max_gamma)
        if cont > 1:
            image = tf.image.random_contrast(image, 1 - cont, 1 + cont)
        if min_jpeg_quality < max_jpeg_quality:
            image = tf.image.random_jpeg_quality(image, min_jpeg_quality,
                                                 max_jpeg_quality)
        if noise > 0:
            image = image + tf.cast(
                tf.random.uniform(shape=[input_shape[1], input_shape[0], 3],
                                  minval=0,
                                  maxval=noise), tf.float32)
        if blur:
            image = random_blur(image)
    else:
        nh = ih * tf.minimum(w / iw, h / ih)
        nw = iw * tf.minimum(w / iw, h / ih)
        dx = (w - nw) / 2
        dy = (h - nh) / 2
        image = tf.image.resize(image,
                                [tf.cast(nh, tf.int32),
                                 tf.cast(nw, tf.int32)])
        new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32),
                                                 tf.cast(dx, tf.int32),
                                                 tf.cast(h, tf.int32),
                                                 tf.cast(w, tf.int32))
        image_color_padded = tf.cast(tf.equal(new_image, 0),
                                     tf.float32) * (128 / 255)
        image = image_color_padded + new_image
        xmin = xmin * nw / iw + dx
        xmax = xmax * nw / iw + dx
        ymin = ymin * nh / ih + dy
        ymax = ymax * nh / ih + dy
    bbox = tf.concat([xmin, ymin, xmax, ymax, tf.cast(label, tf.float32)], 0)
    bbox = tf.transpose(bbox, [1, 0])
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    bbox = tf.clip_by_value(bbox,
                            clip_value_min=0,
                            clip_value_max=tf.cast(input_shape[0] - 1,
                                                   tf.float32))
    bbox_w = bbox[..., 2] - bbox[..., 0]
    bbox_h = bbox[..., 3] - bbox[..., 1]
    bbox = tf.boolean_mask(bbox, tf.logical_and(bbox_w > 1, bbox_h > 1))
    bbox = tf.cond(tf.greater(
        tf.shape(bbox)[0], max_boxes), lambda: bbox[:max_boxes], lambda: bbox)

    return image, bbox
