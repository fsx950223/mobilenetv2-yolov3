import tensorflow as tf

# class FreezableBatchNorm(tf.keras.layers.BatchNormalization):
#   """Batch normalization layer (Ioffe and Szegedy, 2014).
#
#   This is a `freezable` batch norm layer that supports setting the `training`
#   parameter in the __init__ method rather than having to set it either via
#   the Keras learning phase or via the `call` method parameter. This layer will
#   forward all other parameters to the default Keras `BatchNormalization`
#   layer
#
#   This is class is necessary because Object Detection model training sometimes
#   requires batch normalization layers to be `frozen` and used as if it was
#   evaluation time, despite still training (and potentially using dropout layers)
#
#   Like the default Keras BatchNormalization layer, this will normalize the
#   activations of the previous layer at each batch,
#   i.e. applies a transformation that maintains the mean activation
#   close to 0 and the activation standard deviation close to 1.
#
#   Arguments:
#     training: Boolean or None. If True, the batch normalization layer will
#       normalize the input batch using the batch mean and standard deviation,
#       and update the total moving mean and standard deviations. If False, the
#       layer will normalize using the moving average and std. dev, without
#       updating the learned avg and std. dev.
#       If None, the layer will follow the keras BatchNormalization layer
#       strategy of checking the Keras learning phase at `call` time to decide
#       what to do.
#     **kwargs: The keyword arguments to forward to the keras BatchNormalization
#         layer constructor.
#
#   Input shape:
#       Arbitrary. Use the keyword argument `input_shape`
#       (tuple of integers, does not include the samples axis)
#       when using this layer as the first layer in a model.
#
#   Output shape:
#       Same shape as input.
#
#   References:
#       - [Batch Normalization: Accelerating Deep Network Training by Reducing
#         Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
#   """
#
#   def __init__(self, training=None, **kwargs):
#     super(FreezableBatchNorm, self).__init__(**kwargs)
#     self._training = training
#
#   def call(self, inputs, training=None):
#     if training is None:
#       training = self._training
#     return super(FreezableBatchNorm, self).call(inputs, training=training)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


class _LayersOverride:

    def __init__(self,
                 default_batchnorm_momentum=0.999,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None):
        """Alternative tf.keras.layers interface, for use by the Keras MobileNetV2.

        It is used by the Keras applications kwargs injection API to
        modify the Mobilenet v2 Keras application with changes required by
        the Object Detection API.

        These injected interfaces make the following changes to the network:

        - Applies the Object Detection hyperparameter configuration
        - Supports FreezableBatchNorms
        - Adds support for a min number of filters for each layer
        - Makes the `alpha` parameter affect the final convolution block even if it
            is less than 1.0
        - Adds support for explicit padding of convolutions

        Args:
          batchnorm_training: Bool. Assigned to Batch norm layer `training` param
            when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
          default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
            batch norm layers will be constructed using this value as the momentum.
          conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
            containing hyperparameters for convolution ops. Optionally set to `None`
            to use default mobilenet_v2 layer builders.
          use_explicit_padding: If True, use 'valid' padding for convolutions,
            but explicitly pre-pads inputs so that the output dimensions are the
            same as if 'same' padding were used. Off by default.
          alpha: The width multiplier referenced in the MobileNetV2 paper. It
            modifies the number of filters in each convolutional layer.
          min_depth: Minimum number of filters in the convolutional layers.
        """
        self._default_batchnorm_momentum = default_batchnorm_momentum
        self._conv_hyperparams = conv_hyperparams
        self._use_explicit_padding = use_explicit_padding
        self._alpha = alpha
        self._min_depth = min_depth
        self._regularizer = tf.keras.regularizers.l2(0.00004)
        self._initializer = tf.random_normal_initializer(stddev=0.03)

    # def _FixedPaddingLayer(self,kernel_size):
    #     return tf.keras.layers.Lambda(lambda x: _fixed_padding(x, kernel_size))

    # def Conv2D(self,filters,**kwargs):
    #     """Builds a Conv2D layer according to the current Object Detection config.
    #
    #     Overrides the Keras MobileNetV2 application's convolutions with ones that
    #     follow the spec specified by the Object Detection hyperparameters.
    #
    #     Args:
    #       filters: The number of filters to use for the convolution.
    #       **kwargs: Keyword args specified by the Keras application for
    #         constructing the convolution.
    #
    #     Returns:
    #       A one-arg callable that will either directly apply a Keras Conv2D layer to
    #       the input argument, or that will first pad the input then apply a Conv2D
    #       layer.
    #     """
    #     if kwargs.get('name')=='Conv_1' and self._alpha<1.0:
    #         filters=_make_divisible(1280*self._alpha,8)
    #
    #     if self._min_depth and (filters<self._min_depth) and not kwargs.get('name').endswith('expand'):
    #         filters=self._min_depth
    #
    #     # if self._conv_hyperparams:
    #     #     kwargs=self._conv_hyperparams.params(**kwargs)
    #     # else:
    #     #     kwargs['kernel_regularizer']=self._regularizer
    #     #     kwargs['kernel_initializer']=self._initializer
    #
    #     kwargs['padding']='same'
    #     kernel_size=kwargs.get('kernel_size')
    #     if self._use_explicit_padding and kernel_size>1:
    #         kwargs['padding']='valid'
    #         def padded_conv(features):
    #             padded_features=self._FixedPaddingLayer(kernel_size)(features)
    #             return tf.keras.layers.Conv2D(filters,**kwargs)(padded_features)
    #         return padded_conv
    #     else:
    #         return tf.keras.layers.Conv2D(filters,**kwargs)

    # def DepthwiseConv2D(self,**kwargs):
    #     """Builds a DepthwiseConv2D according to the Object Detection config.
    #
    #     Overrides the Keras MobileNetV2 application's convolutions with ones that
    #     follow the spec specified by the Object Detection hyperparameters.
    #
    #     Args:
    #       **kwargs: Keyword args specified by the Keras application for
    #         constructing the convolution.
    #
    #     Returns:
    #       A one-arg callable that will either directly apply a Keras DepthwiseConv2D
    #       layer to the input argument, or that will first pad the input then apply
    #       the depthwise convolution.
    #     """
    #     if self._conv_hyperparams:
    #         kwargs=self._conv_hyperparams.params(**kwargs)
    #     else:
    #         kwargs['depthwise_initializer']=self._initializer
    #
    #     kwargs['padding']='same'
    #     kernel_size=kwargs.get('kernel_size')
    #     if self._use_explicit_padding and kernel_size>1:
    #         kwargs['padding']='valid'
    #         def padded_depthwise_conv(features):
    #             padded_features=self._FixedPaddingLayer(kernel_size)(features)
    #             return tf.keras.layers.DepthwiseConv2D(**kwargs)(padded_features)
    #         return padded_depthwise_conv
    #     else:
    #         return tf.keras.layers.DepthwiseConv2D(**kwargs)

    def BatchNormalization(self, **kwargs):
        """Builds a normalization layer.

        Overrides the Keras application batch norm with the norm specified by the
        Object Detection configuration.

        Args:
          **kwargs: Only the name is used, all other params ignored.
            Required for matching `layers.BatchNormalization` calls in the Keras
            application.

        Returns:
          A normalization layer specified by the Object Detection hyperparameter
          configurations.
        """
        name = kwargs.get('name')
        if self._conv_hyperparams:
            return self._conv_hyperparams.build_batch_norm(name=name)
        else:
            return tf.keras.layers.BatchNormalization(
                momentum=self._default_batchnorm_momentum, name=name)

    # def Input(self,shape):
    #     """Builds an Input layer.
    #
    #     Overrides the Keras application Input layer with one that uses a
    #     tf.placeholder_with_default instead of a tf.placeholder. This is necessary
    #     to ensure the application works when run on a TPU.
    #
    #     Args:
    #       shape: The shape for the input layer to use. (Does not include a dimension
    #         for the batch size).
    #     Returns:
    #       An input layer for the specified shape that internally uses a
    #       placeholder_with_default.
    #     """
    #     default_size = 224
    #     default_batch_size = 1
    #     shape = list(shape)
    #     default_shape = [default_size if dim is None else dim for dim in shape]
    #
    #     input_tensor = tf.constant(0.0, shape=[default_batch_size] + default_shape)
    #
    #     placeholder_with_default = tf.placeholder_with_default(
    #         input=input_tensor, shape=[None] + shape)
    #     return tf.keras.layers.Input(tensor=placeholder_with_default)

    # def ReLU(self, *args, **kwargs):
    #     """Builds an activation layer.
    #
    #     Overrides the Keras application ReLU with the activation specified by the
    #     Object Detection configuration.
    #
    #     Args:
    #       *args: Ignored, required to match the `tf.keras.ReLU` interface
    #       **kwargs: Only the name is used,
    #         required to match `tf.keras.ReLU` interface
    #
    #     Returns:
    #       An activation layer specified by the Object Detection hyperparameter
    #       configurations.
    #     """
    #     name = kwargs.get('name')
    #     if self._conv_hyperparams:
    #         return self._conv_hyperparams.build_activation_layer(name=name)
    #     else:
    #         return tf.keras.layers.Lambda(tf.nn.relu6, name=name)

    # def ZeroPadding2D(self, **kwargs):
    #     """Replaces explicit padding in the Keras application with a no-op.
    #
    #     Args:
    #       **kwargs: Ignored, required to match the Keras applications usage.
    #
    #     Returns:
    #       A no-op identity lambda.
    #     """
    #     return lambda x: x

    def __getattr__(self, item):
        return getattr(tf.keras.layers, item)


def mobilenet_v2(default_batchnorm_momentum=0.999,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None,
                 **kwargs):
    """Instantiates the MobileNetV2 architecture, modified for object detection.

      This wraps the MobileNetV2 tensorflow Keras application, but uses the
      Keras application's kwargs-based monkey-patching API to override the Keras
      architecture with the following changes:

      - Changes the default batchnorm momentum to 0.9997
      - Applies the Object Detection hyperparameter configuration
      - Supports FreezableBatchNorms
      - Adds support for a min number of filters for each layer
      - Makes the `alpha` parameter affect the final convolution block even if it
          is less than 1.0
      - Adds support for explicit padding of convolutions
      - Makes the Input layer use a tf.placeholder_with_default instead of a
          tf.placeholder, to work on TPUs.

      Args:
          batchnorm_training: Bool. Assigned to Batch norm layer `training` param
            when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
          default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
            batch norm layers will be constructed using this value as the momentum.
          conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
            containing hyperparameters for convolution ops. Optionally set to `None`
            to use default mobilenet_v2 layer builders.
          use_explicit_padding: If True, use 'valid' padding for convolutions,
            but explicitly pre-pads inputs so that the output dimensions are the
            same as if 'same' padding were used. Off by default.
          alpha: The width multiplier referenced in the MobileNetV2 paper. It
            modifies the number of filters in each convolutional layer.
          min_depth: Minimum number of filters in the convolutional layers.
          **kwargs: Keyword arguments forwarded directly to the
            `tf.keras.applications.MobilenetV2` method that constructs the Keras
            model.

      Returns:
          A Keras model instance.
      """
    layers_override = _LayersOverride(
        default_batchnorm_momentum=default_batchnorm_momentum,
        conv_hyperparams=conv_hyperparams,
        use_explicit_padding=use_explicit_padding,
        min_depth=min_depth,
        alpha=alpha)
    return tf.keras.applications.MobileNetV2(alpha=alpha,
                                             layers=layers_override,
                                             **kwargs)
