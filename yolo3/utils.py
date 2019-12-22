"""Miscellaneous utility functions."""

from functools import reduce
import tensorflow as tf
import numpy as np

def do_giou_calculate(b1, b2, mode='giou'):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
        encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
        in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'],
        decided to calculate giou loss or iou loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0., b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == 'iou':
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan(
        (enclose_area - union_area), enclose_area)
    return giou


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
    return new_image


def random_gamma(image, min, max):
    val = tf.random.uniform([], min, max)
    return tf.image.adjust_gamma(image, val)


def random_blur(image):
    import cv2
    gaussian_blur = lambda image: cv2.GaussianBlur(image.numpy(), (5, 5), 0)
    h, w = image.shape.as_list()[:2]
    image = tf.py_function(gaussian_blur, [image], tf.float32)
    image.set_shape([h, w, 3])
    return image


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors, np.float32).reshape(-1, 2)


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
                    xmins,
                    xmaxs,
                    ymins,
                    ymaxs,
                    labels,
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
    input_shape=tf.keras.backend.get_value(input_shape)
    iw, ih = tf.cast(tf.shape(image)[1],
                     tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
    w, h = tf.cast(input_shape[1], tf.float32), tf.cast(input_shape[0],
                                                        tf.float32)
    xmaxs = tf.expand_dims(xmaxs, 0)
    xmins = tf.expand_dims(xmins, 0)
    ymaxs = tf.expand_dims(ymaxs, 0)
    ymins = tf.expand_dims(ymins, 0)
    labels = tf.expand_dims(labels, 0)
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
            dy_t = tf.cast(tf.math.maximum(-dy, 0), tf.int32)
            dx_t = tf.cast(tf.math.maximum(-dx, 0), tf.int32)
            image = tf.image.crop_to_bounding_box(
                image, dy_t, dx_t,
                tf.math.minimum(tf.cast(h, tf.int32), tf.cast(nh, tf.int32)),
                tf.math.minimum(tf.cast(w, tf.int32), tf.cast(nw, tf.int32)))
            image = tf.image.pad_to_bounding_box(
                image, tf.cast(tf.math.maximum(dy, 0), tf.int32),
                tf.cast(tf.math.maximum(dx, 0), tf.int32), tf.cast(
                    h, tf.int32), tf.cast(w, tf.int32))
            return image

        image = tf.cond(
            tf.logical_or(nw > w, nh > h), lambda: crop_and_pad(image, dx, dy),
            lambda: tf.image.pad_to_bounding_box(
                image, tf.cast(tf.math.maximum(dy, 0), tf.int32),
                tf.cast(tf.math.maximum(dx, 0), tf.int32), tf.cast(
                    h, tf.int32), tf.cast(w, tf.int32)))

        xmins = xmins * nw / iw + dx
        xmaxs = xmaxs * nw / iw + dx
        ymins = ymins * nh / ih + dy
        ymaxs = ymaxs * nh / ih + dy
        if flip:
            image, xmins, xmaxs = tf.cond(
                tf.less(
                    tf.random.uniform([]),
                    0.5), lambda: (tf.image.flip_left_right(image), w - xmaxs, w
                                   - xmins), lambda: (image, xmins, xmaxs))
        if hue > 0:
            image = tf.image.random_hue(image, hue)
        if sat > 0:
            image = tf.image.random_saturation(image, 1 - sat, 1 + sat)
        if val > 0:
            image = tf.image.random_brightness(image, val)
        if min_gamma < max_gamma:
            image = random_gamma(image, min_gamma, max_gamma)
        if cont > 0:
            image = tf.image.random_contrast(image, 1 - cont, 1 + cont)
        if min_jpeg_quality < max_jpeg_quality:
            image = tf.image.random_jpeg_quality(image, min_jpeg_quality,
                                                 max_jpeg_quality)
        if noise > 0:
            image = image + tf.cast(
                tf.random.uniform(shape=[input_shape[0], input_shape[1], 3],
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
        image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32),
                                                 tf.cast(dx, tf.int32),
                                                 tf.cast(h, tf.int32),
                                                 tf.cast(w, tf.int32))
        xmins = xmins * nw / iw + dx
        xmaxs = xmaxs * nw / iw + dx
        ymins = ymins * nh / ih + dy
        ymaxs = ymaxs * nh / ih + dy

    bbox = tf.concat([xmins, ymins, xmaxs, ymaxs,
                      tf.cast(labels, tf.float32)], 0)
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


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, wh, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    grid_shapes = [
        np.round(input_shape / [32, 16, 8][l]).astype(np.int32)
        for l in range(num_layers)
    ]
    y_true = [
        np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(
            anchor_mask[l]), 5 + num_classes),
                 dtype='float32') for l in range(num_layers)
    ]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    wh = boxes_wh[valid_mask]
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes

    anchor_box=tf.stack([anchor_mins[...,1],anchor_mins[...,0],anchor_maxes[...,1],anchor_maxes[...,0]],axis=-1)
    bbox=tf.stack([box_mins[...,1],box_mins[...,0],box_maxes[...,1],box_maxes[...,0]],axis=-1)
    iou=do_giou_calculate(anchor_box,bbox,mode='iou').numpy()

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
        for l in range(num_layers):
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[t, 0] *
                             grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[t, 1] *
                             grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[t, 4].astype('int32')
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1.
                y_true[l][j, i, k, 5 + c] = 1.

    return y_true[0], y_true[1], y_true[2]


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
