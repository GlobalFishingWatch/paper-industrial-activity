"""Build a ConvNet for object classificaiton.

"""
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow_addons.layers as kla
from tensorflow.keras.activations import gelu
from tensorflow.keras.initializers import Constant

from .dataset import DatasetSource  # noqa

CLASSES = ["fishing", "nonfishig"]

# NOTE: If using LayerNorm, remove "momentum" below
NormLayer = kl.BatchNormalization


class ConvNeXtBlock(kl.Layer):
    """Build a ConvNeXt residual block

    ConvNeXt is described in https://arxiv.org/pdf/2201.03545.pdf

    We use "NormLayer" for normalization. In the paper, this was
    LayerNorm, but it doesn't train well in our application,
    so we typically use BatchNorm.

    Args:
        survival_prob (float): If != 1, use stochastic depth.
        scale (int): Increase/reduce inner filters by this factor.
        layer_scale (bool): If True, apply layer scaling.
    """

    def __init__(self, survival_prob=1, kernel_size=7, scale=4, layer_scale=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.survival_prob = survival_prob
        self.layer_scale = layer_scale
        self.scale = scale

    def build(self, input_shape):
        filters = input_shape[-1]  # num channels
        self.dconv = kl.DepthwiseConv2D(kernel_size=self.kernel_size, padding="same")
        self.norm = NormLayer(center=False, scale=False, momentum=0.999)
        self.pconv1 = kl.Conv2D(self.scale * filters, 1, activation=gelu)
        self.pconv2 = kl.Conv2D(filters, 1)
        if self.layer_scale:
            self.lscale = LayerScale()
        if self.survival_prob == 1:
            self.add = kl.Add()
        else:
            self.add = kla.StochasticDepth(self.survival_prob)

    def call(self, x):
        skip = x
        x = self.dconv(x)
        x = self.norm(x)
        x = self.pconv1(x)
        x = self.pconv2(x)
        if self.layer_scale:
            x = self.lscale(x)
        return self.add([skip, x])


class DownsampleBlock(kl.Layer):
    def __init__(self, filters, kernel_size=2, stride=2, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.norm = NormLayer()
        self.conv = kl.Conv2D(filters, kernel_size, stride, padding="same")

    def call(self, x):
        if self.norm_first:
            x = self.norm(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
        return x


class EvidenceBlock(kl.Layer):
    """Output block for evidential loss function.

    Used for multiclass classification.

    Takes the output of the last dense/classification layer
    (the evidence array) and computes the class scores.

    paper: https://tinyurl.com/2p9aweat

    Return:
        class_score (array): Probabilities for each class.

    See:
        evidential_loss
    """

    def __init__(self, units):
        super().__init__()
        self.dense = kl.Dense(units, activation="softplus")

    def call(self, x):
        evidence = self.dense(x)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        return evidence / S  # class_score


class LayerScale(kl.Layer):
    def __init__(self, base_scale=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.base_scale = base_scale

    def build(self, input_shape):
        shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.scale_factors = self.add_weight(
            shape=shape,
            initializer=Constant(self.base_scale),
            trainable=True,
            name="scale_factors",
        )

    def call(self, x):
        return self.scale_factors * x


class FullyConnectedBlock(km.Model):
    """Build a block with fully connected layers.

    Used to process output from mixed-data multi-input.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dense1 = kl.Dense(32, activation=gelu)
        self.drop1 = kl.Dropout(0.5)
        self.dense2 = kl.Dense(16, activation=gelu)
        self.drop2 = kl.Dropout(0.5)

    def call(self, x):
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x


class Backbone(km.Model):
    """Build a ConvNeXt backbone architecture.

    paper: https://arxiv.org/pdf/2201.03545.pdf
    code: https://tinyurl.com/5ykw657h

    It is kept this way to be compatible with the
      multi-input (dual branch) classification model.

    Args:
        depths (list): number of residual blocks per stage.
        dims (list): number of layer filters in each stage.
        kwargs (key=val): passed to the ResNeXt block.
    """

    def __init__(
        self,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        survival_edges=[0.9, 0.8, 0.7, 0.6, 0.5],
        drop_path=False,
        **kwargs
    ):
        super().__init__()

        # NOTE: use build_stage below, with changing survival prob
        # stages = [
        #     # Multiple convnext blocks per stage
        #     km.Sequential([ConvNeXtBlock(**kwargs) for _ in range(depths[i])])
        #     for i in range(4)
        # ]

        if not drop_path:
            survival_edges = np.ones(len(depths) + 1)

        self.depths = depths
        self.dims = dims
        self.survival_edges = survival_edges

        self.dsample1 = DownsampleBlock(dims[0], 4, 4, norm_first=False)
        self.dsample2 = DownsampleBlock(dims[1], 2, 2)
        self.dsample3 = DownsampleBlock(dims[2], 2, 2)
        self.dsample4 = DownsampleBlock(dims[3], 2, 2)
        self.convnext1 = self.build_stage(0, **kwargs)
        self.convnext2 = self.build_stage(1, **kwargs)
        self.convnext3 = self.build_stage(2, **kwargs)
        self.convnext4 = self.build_stage(3, **kwargs)
        self.norm = NormLayer()
        self.activ = gelu
        self.gpool = kl.GlobalAveragePooling2D()

    def call(self, x):
        x = self.dsample1(x)
        x = self.convnext1(x)
        x = self.dsample2(x)
        x = self.convnext2(x)
        x = self.dsample3(x)
        x = self.convnext3(x)
        x = self.dsample4(x)
        x = self.convnext4(x)
        x = self.norm(x)
        x = self.activ(x)
        x = self.gpool(x)
        return x

    def build_stage(self, i, **kwargs):
        """Build stage i as depths[i] x ConvNeXt blocks.

        Linearly decreases survival probability.
        """
        survivals = [None for _ in self.depths]

        for k, n in enumerate(self.depths):
            survivals[k] = np.linspace(
                self.survival_edges[k], self.survival_edges[k + 1], n + 1
            )
        # [print(survivals[i][j]) for j in range(self.depths[i])]

        return km.Sequential(
            [
                ConvNeXtBlock(survival_prob=survivals[i][j], **kwargs)
                for j in range(self.depths[i])
            ]
        )


class ConvNet(km.Model):
    """Build a mixed-data Convolutional Neural Net classifier.

    Inputs:
        (X1, X2) : (11-channel-img, scalar)

    Returns:
        Classification probabilities: [fishing, nonfishing].
    """

    def __init__(self, num_classes=len(CLASSES), **kwargs):
        super().__init__()
        self.cnn = Backbone(**kwargs)
        self.fc = FullyConnectedBlock()
        self.cat = kl.Concatenate()
        self.head = kl.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x, y = inputs  # image, scalar
        x = self.cnn(x)
        z = self.cat([x, y])
        z = self.fc(z)
        return self.head(z)


Model = ConvNet

# Quick test
# >>> python -m classification.fishing.convnext_mixed
# x1 = tf.ones(shape=(16, 160, 160, 11))
# x2 = tf.ones(shape=(16, 1))
#
# model = Model(
#     depths=[3, 3, 9, 3],
#     dims=[256, 512, 1024, 2048],
#     survival_edges=[.9, .9, .9, .9, .9],
#     drop_path=True,
# )
#
# y = model((x1, x2))
# print(model.summary(), y)
