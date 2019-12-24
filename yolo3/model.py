"""YOLO_v3 Model Defined in Keras."""

from yolo3.enums import BOX_LOSS
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from yolo3.utils import compose,do_giou_calculate
from yolo3.override import mobilenet_v2
from yolo3.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D, darknet_body
from yolo3.efficientnet import EfficientNetB4, MBConvBlock, get_model_params, BlockArgs
from yolo3.train import AdvLossModel


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def darknet_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    if not hasattr(inputs, '_keras_history'):
        inputs = tf.keras.layers.Input(tensor=inputs)
    darknet = darknet_body(inputs, include_top=False)
    x, y1 = make_last_layers(darknet.output, 512,
                             num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y1')(y1)
    y2 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y2')(y2)
    y3 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y3')(y3)
    return AdvLossModel(inputs, [y1, y2, y3])


def MobilenetSeparableConv2D(filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    return compose(
        tf.keras.layers.DepthwiseConv2D(kernel_size,
                                        padding=padding,
                                        use_bias=use_bias,
                                        strides=strides),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(filters,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.))


def make_last_layers_mobilenet(x, id, num_filters, out_filters):
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 1) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 1) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 1) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 2) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 2) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 2) + '_relu6'))(x)
    y = compose(
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(out_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)
    return compose(
        tf.keras.layers.Conv2D(last_block_filters,
                               kernel,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.))


def mobilenetv2_yolo_body(inputs, num_anchors, num_classes, alpha=1.0):
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')
    x, y1 = make_last_layers_mobilenet(mobilenetv2.output, 17, 512,
                                       num_anchors * (num_classes + 5))
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([
        x,
        MobilenetConv2D(
            (1, 1), alpha,
            384)(mobilenetv2.get_layer('block_12_project_BN').output)
    ])
    x, y2 = make_last_layers_mobilenet(x, 21, 256,
                                       num_anchors * (num_classes + 5))
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([
        x,
        MobilenetConv2D((1, 1), alpha,
                        128)(mobilenetv2.get_layer('block_5_project_BN').output)
    ])
    x, y3 = make_last_layers_mobilenet(x, 25, 128,
                                       num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)
    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)
    return AdvLossModel(mobilenetv2.inputs, [y1, y2, y3])


def make_last_layers_efficientnet(x, block_args, global_params):
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    num_filters = block_args.input_filters * block_args.expand_ratio
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.))(x)
    y = compose(
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(block_args.output_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def efficientnet_yolo_body(inputs, model_name, num_anchors, **kwargs):
    _, global_params, input_shape = get_model_params(model_name, kwargs)
    num_classes = global_params.num_classes
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    efficientnet = EfficientNetB4(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_shape, input_shape, 3),
                                  input_tensor=inputs)
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=num_anchors * (num_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    x, y1 = make_last_layers_efficientnet(efficientnet.output, block_args,
                                          global_params)
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, efficientnet.get_layer('swish_65').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, global_params)
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, efficientnet.get_layer('swish_29').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, global_params)
    y1 = tf.keras.layers.Reshape(
        (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
    y2 = tf.keras.layers.Reshape(
        (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
    y3 = tf.keras.layers.Reshape(
        (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)

    return AdvLossModel(efficientnet.inputs, [y1, y2, y3])


def yolo_head(feats: tf.Tensor,
              anchors: np.ndarray,
              input_shape: tf.Tensor,
              calc_loss: bool = False
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = tf.shape(feats)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(
        anchors_tensor, feats.dtype) / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    if calc_loss == True:
        return grid, box_xy, box_wh, box_confidence
    box_class_probs = tf.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy: tf.Tensor, box_wh: tf.Tensor,
                       input_shape: tf.Tensor, image_shape) -> tf.Tensor:
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)
    max_shape = tf.maximum(image_shape[0], image_shape[1])
    ratio = image_shape / max_shape
    boxed_shape = input_shape * ratio
    offset = (input_shape - boxed_shape) / 2.
    scale = image_shape / boxed_shape
    box_yx = (box_yx * input_shape - offset) * scale
    box_hw *= input_shape * scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat(
        [
            tf.clip_by_value(box_mins[..., 0:1], 0, image_shape[0]),  # y_min
            tf.clip_by_value(box_mins[..., 1:2], 0, image_shape[1]),  # x_min
            tf.clip_by_value(box_maxes[..., 0:1], 0, image_shape[0]),  # y_max
            tf.clip_by_value(box_maxes[..., 1:2], 0, image_shape[1])  # x_max
        ],
        -1)
    return boxes


def yolo_boxes_and_scores(feats: tf.Tensor, anchors: List[Tuple[float, float]],
                          num_classes: int, input_shape: Tuple[int, int],
                          image_shape) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs: List[tf.Tensor],
              anchors: np.ndarray,
              num_classes: int,
              image_shape,
              max_boxes: int = 20,
              score_threshold: float = .6,
              iou_threshold: float = .5
             ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        nms_index = tf.image.non_max_suppression(
            boxes,
            box_scores[:, c],
            max_boxes_tensor,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold)
        class_boxes = tf.gather(boxes, nms_index)
        class_box_scores = tf.gather(box_scores[:, c], nms_index)
        classes = tf.ones_like(class_box_scores, tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0, name='scores')
    classes_ = tf.concat(classes_, axis=0, name='classes')
    boxes_ = tf.cast(boxes_, tf.int32, name='boxes')
    return boxes_, scores_, classes_


class YoloEval(tf.keras.layers.Layer):

    def __init__(self,
                 anchors,
                 num_classes,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5,
                 **kwargs):
        super(YoloEval, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def call(self, yolo_outputs,image_shape):
        return yolo_eval(yolo_outputs, self.anchors, self.num_classes,
                         image_shape, self.max_boxes, self.score_threshold,
                         self.iou_threshold)

    def get_config(self):
        config = super(YoloEval, self).get_config()
        config['anchors'] = self.anchors
        config['num_classes'] = self.num_classes
        config['max_boxes'] = self.max_boxes
        config['score_threshold'] = self.score_threshold
        config['iou_threshold'] = self.iou_threshold

        return config


class YoloLoss(tf.keras.losses.Loss):

    def __init__(self,
                    idx,
                    anchors,
                    ignore_thresh=.5,
                    box_loss=BOX_LOSS.GIOU,
                    print_loss=True):
        super(YoloLoss, self).__init__(reduction=tf.losses.Reduction.NONE,name='yolo_loss')
        grid_steps = [32, 16, 8]
        anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.idx = idx
        self.ignore_thresh = ignore_thresh
        self.box_loss = box_loss
        self.print_loss = print_loss
        self.grid_step = grid_steps[self.idx]
        self.anchor = anchors[anchor_masks[idx]]

    def call(self, y_true, yolo_output):
        '''Return yolo_loss tensor

        Parameters
        ----------
        yolo_output: the output of yolo_body
        y_true: the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss

        Returns
        -------
        loss: tensor, shape=(1,)

        '''
        loss = 0
        m = tf.shape(yolo_output)[0]  # batch size, tensor
        mf = tf.cast(m, yolo_output.dtype)
        object_mask = y_true[..., 4:5]
        true_class_probs = y_true[..., 5:]
        input_shape = tf.shape(yolo_output)[1:3] * self.grid_step
        grid, pred_xy, pred_wh, box_confidence = yolo_head(
            yolo_output, self.anchor, input_shape, calc_loss=True)
        pred_max = tf.reverse(pred_xy + pred_wh / 2., [-1])
        pred_min = tf.reverse(pred_xy - pred_wh / 2., [-1])
        pred_box = tf.concat([pred_min, pred_max], -1)
        
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_max = tf.reverse(true_xy + true_wh / 2., [-1])
        true_min = tf.reverse(true_xy - true_wh / 2., [-1])
        true_box = tf.concat([true_min, true_max], -1)
        true_box = tf.clip_by_value(true_box, 0, 1)
        object_mask_bool = tf.cast(object_mask, 'bool')

        masked_true_box = tf.boolean_mask(true_box, object_mask_bool[..., 0])
        iou = do_giou_calculate(
            tf.expand_dims(pred_box, -2),
            tf.expand_dims(masked_true_box, 0),
            mode='iou')
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, masked_true_box.dtype)

        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # focal_loss = focal(object_mask, box_confidence)
        confidence_loss = (object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=yolo_output[..., 4:5]) + \
                        (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                    logits=yolo_output[...,
                                                                                            4:5]) * ignore_mask)
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_class_probs, logits=yolo_output[..., 5:])
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        if self.box_loss == BOX_LOSS.GIOU:
            giou = do_giou_calculate(pred_box, true_box)
            giou_loss = object_mask * (1 - tf.expand_dims(giou, -1))
            giou_loss = tf.reduce_sum(giou_loss) / mf
            loss += giou_loss + confidence_loss + class_loss
            if self.print_loss:
                tf.print(str(self.idx)+':',giou_loss, confidence_loss, class_loss,tf.reduce_sum(ignore_mask))
        elif self.box_loss == BOX_LOSS.MSE:
            grid_shape = tf.cast(tf.shape(yolo_output)[1:3], y_true.dtype)
            raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
            raw_true_wh = tf.math.log(y_true[..., 2:4] /
                                    tf.cast(anchors[anchor_mask[idx]], y_true.dtype)*
                                    tf.cast(input_shape[::-1], y_true.dtype) )
            raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh,
                                                tf.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=raw_true_xy, logits=yolo_output[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(
                raw_true_wh - yolo_output[..., 2:4])
            xy_loss = tf.reduce_sum(xy_loss) / mf
            wh_loss = tf.reduce_sum(wh_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
            if print_loss:
                tf.print(loss, xy_loss, wh_loss, confidence_loss, class_loss,
                        tf.reduce_sum(ignore_mask))
        return loss