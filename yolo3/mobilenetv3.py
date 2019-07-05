import tensorflow as tf

class Bneck(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 expansion_filters,
                 kernel_size,
                 alpha=1.0,
                 strides=(1, 1),
                 use_se=False,
                 activation=tf.nn.relu6,
                 **kwargs):
        super(Bneck, self).__init__(**kwargs)
        self.filters = _make_divisible(filters * alpha, 8)
        self.expansion_filters = expansion_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_se = use_se
        self.activation = activation
        self.expand_conv2d = tf.keras.layers.Conv2D(self.expansion_filters, 1, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.expand_bn = tf.keras.layers.BatchNormalization()
        self.zero_padding2d = tf.keras.layers.ZeroPadding2D(((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        self.depthwise_conv2d = tf.keras.layers.DepthwiseConv2D(self.kernel_size, strides=self.strides, use_bias=False,
                                                                padding='same' if self.strides == 1 else 'valid',depthwise_regularizer=tf.keras.regularizers.l2(1e-5))
        self.depthwise_bn = tf.keras.layers.BatchNormalization()
        self.se = SeBlock()
        self.project_conv2d = tf.keras.layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.project_bn = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.built = True

    def call(self, inputs):
        x = self.expand_conv2d(inputs)
        x = self.expand_bn(x)
        x = self.activation(x)
        if self.strides == 2:
            x = self.zero_padding2d(x)
        x = self.depthwise_conv2d(x)
        x = self.depthwise_bn(x)
        if self.use_se:
            x = self.se(x)
        x = self.activation(x)
        x = self.project_conv2d(x)
        x = self.project_bn(x)
        if self.in_channels == self.filters and self.strides == 1:
            x = self.add([inputs, x])
        return x


class SeBlock(tf.keras.layers.Layer):
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.average_pool = tf.keras.layers.AveragePooling2D((int(input_shape[1]),int(input_shape[2])))
        self.dense1 = tf.keras.layers.Dense(int(input_shape[-1]) // self.reduction, use_bias=False,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(int(input_shape[-1]), use_bias=False,activation=tf.keras.activations.hard_sigmoid)
        self.built = True

    def call(self, inputs):
        x = self.average_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs *x

def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6

class HSwish(tf.keras.layers.Layer):
  def call(self, inputs):
    return h_swish(inputs)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def MobilenetV3(input_shape,num_classes, size="large", include_top=True,alpha=1.0):
    input = tf.keras.layers.Input([*input_shape, 3])
    first_block_filters = _make_divisible(16 * alpha, 8)
    if size not in ['large', 'small']:
        raise ValueError('size should be large or small')
    if size == "large":
        x = tf.keras.layers.Conv2D(first_block_filters, 3, strides=2, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = HSwish()(x)
        x = Bneck(16, 16, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 64, 3, alpha=alpha, strides=2, use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 72, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(40, 72, 5, alpha=alpha, strides=2, use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(40, 120, 5, alpha=alpha, strides=1, use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(40, 120, 5, alpha=alpha, strides=1, use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(80, 240, 3, alpha=alpha, strides=2, use_se=False, activation=h_swish)(x)
        x = Bneck(80, 200, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)(x)
        x = Bneck(80, 184, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)(x)
        x = Bneck(80, 184, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)(x)
        x = Bneck(112, 480, 3, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(112, 672, 3, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(160, 672, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)(x)
        x = Bneck(160, 960, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(160, 960, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = tf.keras.layers.Conv2D(_make_divisible(960 * alpha, 8), 1, use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = HSwish()(x)
    else:
        x = tf.keras.layers.Conv2D(first_block_filters, 3, strides=2, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = HSwish()(x)
        x = Bneck(16, 16, 3, alpha=alpha, strides=2, use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(24, 72, 3, alpha=alpha, strides=2, use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 88, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(40, 96, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)(x)
        x = Bneck(40, 240, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(40, 240, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(48, 120, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(48, 144, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(96, 288, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)(x)
        x = Bneck(96, 576, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = Bneck(96, 576, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)(x)
        x = tf.keras.layers.Conv2D(_make_divisible(576 * alpha, 8), 1, use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x=SeBlock()(x)
        output = HSwish()(x)
    if include_top:
        output = tf.keras.layers.AveragePooling2D(pool_size=x.shape[1:3])(output)
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
        output = tf.keras.layers.Conv2D(last_block_filters,1, use_bias=False,activation=h_swish,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(output)
        output = tf.keras.layers.Dropout(0.8)(output)
        output = tf.keras.layers.Conv2D(num_classes,1, use_bias=True,activation=tf.keras.activations.softmax,kernel_regularizer=tf.keras.regularizers.l2(1e-5))(output)
        output = tf.keras.layers.Flatten()(output)
    return tf.keras.Model(input,output)
