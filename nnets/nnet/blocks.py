import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers as kreg
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Cropping2D, Dense,
                                     GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Lambda, MaxPool2D,
                                     Permute, ReLU, Reshape, Softmax, multiply)

from ..nnet.coordconv import CoordConv2D
from ..nnet.deformconv import ConvOffset2D
from ..nnet.dropblock import DropBlock2D

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import custom_gradient
# from tensorflow.python.util import dispatch


# FIXME: Unable to serialize the model, using swish instead
# @dispatch.add_dispatch_support
# @custom_gradient.custom_gradient
# def mish(y):
#     """Mish activation function
#
#     See: https://arxiv.org/abs/1908.08681
#
#     Based on the Tensorflow implementation of SiLU
#     (AKA Swish) found here:
#     https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/ops/nn_impl.py#L512-L551
#     """
#
#     y = ops.convert_to_tensor(y, name="features")
#
#     def grad(dy):
#         """Gradient for the Mish activation function"""
#         # Naively, requires keeping extra stuff
#         # around for backprop, increasing the tensor's memory consumption.
#         # We use a control dependency here so that stuff is re-computed
#         # during backprop (the control dep prevents it being de-duped with the
#         # forward pass) and we can free the sigmoid(features) expression immediately
#         # after use during the forward pass.
#         with ops.control_dependencies([dy]):
#             T = tf.math.tanh(tf.math.softplus(y))
#             S = tf.nn.swish(y)
#             D = 1 - T ** 2
#         activation_grad = D * S + T
#         return dy * activation_grad
#
#     return y * tf.math.tanh(tf.math.softplus(y)), grad
#
#
# default_activation = Lambda(mish)
default_activation = swish


def add_channel_attn(tensor, ratio=16):
    """Add a squeeze excite block to the top of a network.

    See https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    tensor : Keras tensor
    ratio : int, optional
        Higher number make the block simpler: less costly,
        less powerful, and probably less prone to overfitting.

    Returns
    -------
    Keras tensor
    """
    # Initially from https://github.com/titu1994/keras-squeeze-excite-network
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 2 * filters)

    sea = GlobalAveragePooling2D()(init)
    sem = GlobalMaxPooling2D()(init)
    se = Concatenate()([sea, sem])

    se = Reshape(se_shape)(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)

    if K.image_data_format() == "channels_first":
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def add_spatial_attn(input_feature, dilation_rate=1):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    # Reproject features so that we chose correct combo to key off of.
    cbam_feature = Conv2D(
        filters=channel, kernel_size=1, activation=None, padding="same"
    )(cbam_feature)

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        dilation_rate=dilation_rate,
    )(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def merge_attention_global(inputs, ratio=8):
    """Merge and/or gate inputs

    TODO: describe

    Parameters
    ----------
    inputs : Keras layer
    ratio : int, optional
        Features are down sampled by this factor before
        computing attention

    Returns
    -------
    Keras layer
    """

    if len(inputs) == 1:
        [consensus] = inputs
    else:
        consensus = Add()(inputs)
    n_filters = consensus.shape[-1]

    gap = GlobalAveragePooling2D()(consensus)
    gmp = GlobalMaxPooling2D()(consensus)
    gp = Concatenate()([gap, gmp])

    y = Dense(n_filters // ratio, use_bias=False)(gp)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    args = [Dense(n_filters)(y) for _ in range(len(inputs) + 1)]
    args = [Reshape((n_filters, 1))(x) for x in args]
    args = Concatenate(axis=-1)(args)
    mix = Softmax()(args)
    mix = Reshape((1, 1, n_filters, len(inputs) + 1))(mix)
    outputs = [
        multiply([mix[:, :, :, :, i], x]) for (i, x) in enumerate(inputs)
    ]

    if len(inputs) == 1:
        [output] = outputs
    else:
        output = Add()(outputs)

    return output


def merge_attention_local(inputs, ratio=8):
    """"""

    if len(inputs) == 1:
        [consensus] = inputs
    else:
        consensus = Add()(inputs)
    h, w, n_filters = consensus.shape[1:]

    y = Conv2D(n_filters // ratio, 1, use_bias=False)(consensus)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    args = [Conv2D(n_filters, 1)(y) for _ in range(len(inputs) + 1)]
    args = [Reshape((h, w, n_filters, 1))(x) for x in args]
    args = Concatenate(axis=-1)(args)
    mix = Softmax()(args)
    mix = Reshape((h, w, n_filters, len(inputs) + 1))(mix)
    outputs = [
        multiply([mix[:, :, :, :, i], x]) for (i, x) in enumerate(inputs)
    ]

    if len(inputs) == 1:
        [output] = outputs
    else:
        output = Add()(outputs)

    return output


def add_res_block(
    inputs,
    filter_count,
    kernel_size=3,
    stride=1,
    l2=0,
    keep_prob=1,
    scale=4,
    activation=default_activation,
    defconv=False,
    groups=1,
    dilation_rate=1,
):
    """Build a ResNeSt residual block

    https://arxiv.org/abs/2004.08955

    Parameters
    ----------
    TODO:

    Returns
    -------
    Keras tensor
    """

    assert not (groups > 1 and defconv)

    kr = kreg.l2(l2)
    input_filter_count = inputs.shape[-1]

    project_shortcut = (input_filter_count != filter_count) | (stride != 1)

    if project_shortcut:
        y0 = Conv2D(
            filter_count,
            stride,
            strides=stride,
            padding="same",
            kernel_regularizer=kr,
        )(inputs)
    else:
        y0 = inputs

    inner_filter_count = filter_count // scale

    def make_branch(y1):
        y1 = BatchNormalization()(y1)
        y1 = activation(y1)
        y1 = Conv2D(inner_filter_count, 1, kernel_regularizer=kr)(y1)
        y1 = BatchNormalization()(y1)
        y1 = activation(y1)
        if keep_prob < 1:
            y1 = DropBlock2D(5, keep_prob)(y1)
        if defconv:
            y1 = ConvOffset2D(filters=inner_filter_count, strides=stride)(y1)
        else:
            y1 = Conv2D(
                inner_filter_count,
                kernel_size,
                kernel_regularizer=kr,
                strides=stride,
                padding="same",
                groups=groups,
                dilation_rate=dilation_rate,
            )(y1)
        y1 = BatchNormalization()(y1)
        y1 = activation(y1)
        y1 = Conv2D(filter_count, 1, kernel_regularizer=kr)(y1)
        return y1

    dy = make_branch(inputs)

    return Add()([dy, y0])


def add_simple_block(
    y,
    filter_count,
    kernel_size=3,
    stride=1,
    l2=0,
    groups=1,
    keep_prob=1,
    batch_norm=True,
    activation=default_activation,
    dilation_rate=1,
    cconv=False,
):
    kr = kreg.l2(l2)
    if keep_prob < 1:
        y = DropBlock2D(5, keep_prob)(y)
    if cconv:
        assert groups == 1
        y = CoordConv2D(
            filter_count,
            kernel_size,
            strides=stride,
            kernel_regularizer=kr,
            dilation_rate=dilation_rate,
            padding="same",
        )(y)
    else:
        y = Conv2D(
            filter_count,
            kernel_size,
            strides=stride,
            groups=groups,
            kernel_regularizer=kr,
            dilation_rate=dilation_rate,
            padding="same",
        )(y)
    if batch_norm:
        y = BatchNormalization()(y)
    if activation is not None:
        y = activation(y)
    return y


# NOTE: Moved here from detection_model


def add_conv_block(y, keep_prob=1, l2=0, n_filters=None):
    """Conv Block from PP-YOLO Figure 2"""
    if n_filters is None:
        n_filters = y.shape[-1]
    y = add_simple_block(y, 2 * n_filters, 3, l2=l2)
    if keep_prob < 1:
        y = DropBlock2D(5, keep_prob)(y)
    return add_simple_block(y, n_filters, 1, l2=l2)  # cconv=True,


def add_upsample_block(y, l2=0):
    """Upsample Block from PP-YOLO Figure 2"""
    n_filters = y.shape[-1]
    y = add_simple_block(y, n_filters // 2, 1, l2=l2)  # cconv=True,
    return tf.keras.layers.UpSampling2D(2, interpolation="nearest")(y)


def add_fpn_block(
    y, prev, filters, keep_prob, fppool=False, l2=0, dilation_rate=1
):
    """One of the blocks comprising the FPN in PP-YOLO Figure 2"""
    if prev is not None:
        prev = add_upsample_block(prev, l2=l2)
        y = Concatenate()([y, prev])
    y = add_simple_block(y, filters, 1, l2=l2)  # cconv=True,
    y = add_conv_block(y, keep_prob=keep_prob, l2=l2)
    if fppool:
        p1 = y
        p5 = MaxPool2D(5, strides=1, padding="same")(y)
        p9 = MaxPool2D(9, strides=1, padding="same")(y)
        p13 = MaxPool2D(13, strides=1, padding="same")(y)
        y = Concatenate()([p1, p5, p9, p13])
    y = add_conv_block(y, l2=l2, n_filters=filters)
    y = add_channel_attn(y)
    y = add_spatial_attn(y, dilation_rate=dilation_rate)
    return y


# NOTE: Original
# def add_output_block(
#     y, scale, max_length, max_speed=30, eps=0.05, crop_0=32, l2=0
# ):
#     n_filters = y.shape[-1]
#     kr = kreg.l2(l2)
#     name = f"Y{scale}0"
#     crop = crop_0 // scale
#     y = add_simple_block(y, 2 * n_filters, 3, l2=l2)
#     region = Conv2D(8, 1, kernel_regularizer=kr)(y)
#     orient = Lambda(
#         lambda x: 0.5 * tf.atan2(x[:, :, :, 4:5], x[:, :, :, 5:6]),
#         name=f"orient_{scale}",
#     )(region)
#     hv = Lambda(
#         lambda x: (1 + eps) * max_length * tf.tanh(x[:, :, :, 0:2]),
#         name=f"hv_{scale}",
#     )(region)
#     l = Lambda(
#         lambda x: (1 + eps) * max_length * tf.sigmoid(x[:, :, :, 2:3]),
#         name=f"l_{scale}",
#     )(region)
#     s = Lambda(
#         lambda x: (1 + eps) * max_speed * tf.sigmoid(x[:, :, :, 3:4]),
#         name=f"s_{scale}",
#     )(region)
#     iou = Lambda(lambda x: tf.sigmoid(x[:, :, :, 6:7]), name=f"iou_{scale}")(
#         region
#     )
#     presence = Conv2D(1, 1, activation="sigmoid", kernel_regularizer=kr)(y)
#     output = Concatenate()([presence, hv, l, s, orient, iou])
#     return Cropping2D(crop, name=name)(output)


def add_output_block(y, max_length=480.0, eps=0.05):
    """ Multi output.

    Returns:
        classification: presence (score 0-1)
        regression: length (0-480m)

    """

    def scaled_sigmoid(x):
        return (1 + eps) * max_length * tf.sigmoid(x)

    presence = Dense(1, activation="sigmoid", name="presence")(y)
    length = Dense(1, activation=scaled_sigmoid, name="length")(y)
    return [presence, length]


def add_output_block2(y, max_length=480.0, eps=0.05):
    """
    Returns:
        classification: presence (prob 0-1)
        regression: length (0-480m)

    Notes:
        Adds shim layers prior to classification and regression:

        FINAL_EMBEDDING -+-> Dense(256) -> Dense(1)  # Presence
                     |
                     +-> Dense(256) -> Dense(1)  # Length

    """

    def scaled_sigmoid(x):
        return (1 + eps) * max_length * tf.sigmoid(x)

    y1 = Dense(256, activation=swish)(y)
    presence = Dense(1, activation="sigmoid", name="presence")(y1)

    y2 = Dense(256, activation=swish)(y)
    length = Dense(1, activation=scaled_sigmoid, name="length")(y2)
    return [presence, length]


def add_evidence_block(y, activation='softplus', n_classes=5):
    """Output block for evidential loss function.

    It takes the output of the last dense layer
    (the evidence array) and computes the class scores.

    Used for multi-class classification.

    Return
    ------
    class_score: array of float
        Probabilities for each class.

    Paper: https://tinyurl.com/2p9aweat

    """
    evidence = Dense(n_classes, activation=activation, name="evidence")(y)
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    class_score = evidence / S
    return class_score
