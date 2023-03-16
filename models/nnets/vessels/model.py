"""Build a vessel/noise classification model

build_model
    Make a Keras model for boat detection
squared_error_loss
    Loss function for length target
squared_error_metric
    Metric function for length while training

Notes:
    - Input layers would have fewer channels.
    - The FPN layers would be dropped.
    - A global pooling layer would be added.
    - The final layers would be different since you’re
      predicting just two scalars, not a big pile of images.
    - And the loss get’s simplified a lot.

"""
from ..nnet.blocks import add_res_block
from ..nnet.blocks import add_simple_block
from ..nnet.blocks import add_output_block
from ..nnet.blocks import swish
# from .nnet_blocks import mish  # FIXME: Unable to serialize the model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D

# VH, VV
channels = 2


def squared_error(y_true, y_pred):
    """Masked squared error loss/metric for length.

    Mask loss when length=0 (presence=0 -> no vessel).
    """
    mask = tf.cast(y_true > 0, "float32")
    return mask * (y_true - y_pred) ** 2


# NOTE: Remove keep_prob ???
def build_model(
    base_filter_count, tile_size=None, l2=0, keep_prob=1.0, groups=32
):
    """Create a net suitable for object classification.

    We use the (ResNet) backbone from PP-YOLO
    (https://arxiv.org/abs/2007.12099), which is in turn heavily
    influenced by YOLOv3 (https://arxiv.org/abs/1804.02767)
    and ResNet-50 (https://arxiv.org/abs/1512.03385).

    The output block is a standard global pooling plus
    dense layer for single-class output.

    Parameters
    ----------
    base_filter_count : int
        Number of filters used by layers in initial layers of the net.
    tile_size : int or None, optional
        Size in pixels of the input tiles to the net. Layers are
        Assumed to be square. If None, the layers can have variable
        size as long as the correct proportions are maintained.
    l2 : float, optional
        Amount of l2 normalization to add to the net.
    keep_prob : float, optional
        Passed to DropBlock
    groups : int
        if > 1, activate grouped convolutions (only on GPU)

    """
    y = input_layer = Input(shape=(tile_size, tile_size, channels))

    y = add_simple_block(y, base_filter_count // 2, 7, l2=l2, stride=2, activation=swish)

    yp = MaxPool2D(3, strides=2, padding='same')(y)
    yp = add_simple_block(yp, base_filter_count // 2, 1, l2=l2, activation=None)
    yc = add_simple_block(y, base_filter_count // 2, 3, l2=l2, stride=2, activation=None)
    y = Concatenate()([yp, yc])

    y = add_res_block(y, 4 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 4 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 4 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 4 * base_filter_count, groups=groups, scale=2, l2=l2)

    y = add_res_block(y, 8 * base_filter_count, groups=groups, scale=2, l2=l2, stride=2)
    y = add_res_block(y, 8 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 8 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 8 * base_filter_count, groups=groups, scale=2, l2=l2)

    y = add_res_block(y, 16 * base_filter_count, groups=groups, scale=2, l2=l2, stride=2)
    y = add_res_block(y, 16 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 16 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 16 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 16 * base_filter_count, groups=groups, scale=2, l2=l2)

    y = add_res_block(y, 32 * base_filter_count, groups=groups, scale=2, l2=l2, stride=2)
    y = add_res_block(y, 32 * base_filter_count, groups=groups, scale=2, l2=l2)
    y = add_res_block(y, 32 * base_filter_count, groups=groups, scale=2, l2=l2)

    y = swish(y)
    y = GlobalAveragePooling2D()(y)

    # y -> [presense, length]
    outputs = add_output_block(y, max_length=480.0, eps=0.05)
    inputs = (input_layer,)

    return Model(inputs=inputs, outputs=outputs)
