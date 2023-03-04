"""Build a ConvNet for object classificaiton.

"""
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow_addons.layers as kla
from tensorflow.keras.activations import gelu
from tensorflow.keras.initializers import Constant

from ..nnet.losses import evidential_loss as _evidential_loss
from ..nnet.metrics import categorical_accuracy
from .dataset import DatasetSource  # noqa


CLASSES = ["wind", "oil", "other", "noise"]

NormLayer = kl.BatchNormalization
# NormLayer = kl.LayerNormalization  # doesn't work


class ConvNeXtBlock(kl.Layer):
    """Build a ConvNeXt residual block

    ConvNeXt is described in https://arxiv.org/pdf/2201.03545.pdf

    We use *NormLayer* for normalization. In the paper, that was
    NormLayer, but that does not train well in our application,
    so we use BatchNorm instead.

    Args:
        survival_prob (float): If != 1, use stochastic depth.
        scale (int): Increase/reduce inner filters by this factor.
        layer_scale (bool): If True, apply layer scaling.
    """

    def __init__(
        self, survival_prob=1, kernel_size=7, scale=4, layer_scale=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.survival_prob = survival_prob
        self.layer_scale = layer_scale
        self.scale = scale

    def build(self, input_shape):
        filters = input_shape[-1]  # channels
        self.dconv = kl.DepthwiseConv2D(
            kernel_size=self.kernel_size, padding="same"
        )
        self.norm = NormLayer(center=False, scale=False)
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
        self.dense = kl.Dense(units, activation='softplus')

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


class Backbone(km.Model):
    """Build a ConvNeXt backbone architecture.

    paper: https://arxiv.org/pdf/2201.03545.pdf
    code: https://tinyurl.com/5ykw657h

    Replace original classification head with activation and
    global pooling for mergin multi outputs later on.

    Args:
        depths (list): number of residual blocks per stage.
        dims (list): number of layer filters in each stage.
        kwargs (key=val): passed to the ResNeXt block.
    """

    def __init__(
        self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
    ):
        super().__init__()

        stages = [
            # Multiple convnext blocks per stage
            km.Sequential([ConvNeXtBlock(**kwargs) for _ in range(depths[i])])
            for i in range(4)
        ]

        self.dsample1 = DownsampleBlock(dims[0], 4, 4, norm_first=False)
        self.dsample2 = DownsampleBlock(dims[1], 2, 2)
        self.dsample3 = DownsampleBlock(dims[2], 2, 2)
        self.dsample4 = DownsampleBlock(dims[3], 2, 2)
        self.convnext1 = stages[0]
        self.convnext2 = stages[1]
        self.convnext3 = stages[2]
        self.convnext4 = stages[3]
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


class ConvNet(km.Model):
    """Build a Convolutional Neural Net classifier.

    Multi input tensor. Two multichannel imgs: SAR and Optical.
    Custom evidential loss (with non-trainable variable), optional.
    Pass each input through a ConvNeXt and concatenate the outputs.

    Returns:
        Custom class_score from evidence (with softplus), optional.
        Standard classification probability (with softmax), optional.
    """

    def __init__(
        self,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=len(CLASSES),
        use_evidence=True,
        **kwargs
    ):
        super().__init__()
        self.branch1 = Backbone(depths, dims, **kwargs)
        self.branch2 = Backbone(depths, dims, **kwargs)
        self.concat = kl.Concatenate()
        self.drop = kl.Dropout(0.5)
        self.dense = kl.Dense(256, activation=gelu)
        if use_evidence:
            self.head = EvidenceBlock(num_classes)
        else:
            self.head = kl.Dense(num_classes, activation='softmax')

        self.lambda_t = tf.Variable(1e-3, trainable=False)

    def call(self, inputs):
        x, y = inputs
        x = self.branch1(x)
        y = self.branch2(y)
        z = self.concat([x, y])
        z = self.drop(z)
        z = self.dense(z)
        return self.head(z)

    def evidential_loss(self):
        # Redefine loss here so it has access to the above variable
        def loss(y_true, y_pred):
            return _evidential_loss(y_true, y_pred, lambda_t=self.lambda_t)

        return loss

    def evidential_metric(self):
        return self.evidential_loss()

    def accuracy_metric(self):
        return categorical_accuracy


Model = ConvNet

# Quick test
# python -m classification.infra.convnext_multi
# x1 = tf.ones(shape=(0, 100, 100, 2))
# x2 = tf.ones(shape=(0, 100, 100, 4))
# model = Model(depths=[3, 3, 9, 3], dims=[256, 512, 1024, 2048])
# y = model([x1, x2])
# print(model.summary(), y)
