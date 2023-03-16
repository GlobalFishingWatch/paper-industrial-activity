from typing import Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as kl
from tensorflow.keras.activations import gelu

import tensorflow_probability as tfp


class Swag(kl.Layer):
    """Build a Swag residual layer"""

    rng_seed = (8, 88)
    core_layer = kl.Conv2D

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        groups: int = 8,
        alpha: float = 2.0,
        activation=None,
        input_groups=None,
        **kwargs
    ):
        """Initialize Swag layer
        Args:
            filters
            kernel_size
            scale (optional): Factor to increase inner filters by.
            groups (optional): Number of output groups (heads) to use
            alpha (optional): Controls how the groups are sampled during
                training. One corresponds to a uniform distribution, while
                larger values tend to sample primarily from one group
                at a time and smaller values tend to sample all groups
                simultaneously.
            activation (optional)
            input_groups (optional)
            kwargs: Keyword arguments passed onto internal Conv2D layer.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.input_groups = groups if (input_groups is None) else input_groups
        self.alpha = alpha
        self.activation = activation
        self.kwargs = kwargs
        assert (
            self.groups % self.input_groups == 0
        ), "input_groups must evenly divide groups"
        assert self.filters % self.groups == 0, "groups must evenly divide filters"

    def build(self, input_shape):
        self.input_filters = input_shape[-1]
        assert (
            self.input_filters % self.input_groups == 0
        ), "input groups must evenly divide input filters"

        self.conv_lyr = self.core_layer(
            filters=(self.groups // self.input_groups) * self.filters,
            kernel_size=self.kernel_size,
            **self.kwargs
        )
        self.alpha_vector = tf.Variable(
            [self.alpha] * self.groups, name="alpha", trainable=False, dtype="float32"
        )
        self.sample_shape = (input_shape[0], 1, 1)
        self.dirichlet = tfp.distributions.Dirichlet(self.alpha_vector)

    def call(self, x, training=None):
        shpA = tf.shape(x)
        shp1 = tf.concat(
            [shpA[:-1], (self.input_groups, self.input_filters // self.input_groups)],
            axis=0,
        )
        x = tf.reshape(x, shp1)
        x = tf.transpose(x, (0, 3, 1, 2, 4))
        shp2 = tf.concat(
            [
                (shpA[0] * self.input_groups,),
                shpA[1:-1],
                (self.input_filters // self.input_groups,),
            ],
            axis=0,
        )
        x = tf.reshape(x, shp2)
        x = self.conv_lyr(x)
        shpB = tf.shape(x)
        shp3 = tf.concat(
            [
                (
                    shpA[0],
                    self.input_groups,
                ),
                shpB[1:],
            ],
            axis=0,
        )
        x = tf.reshape(x, shp3)
        x = tf.transpose(x, (0, 2, 3, 1, 4))
        shpC = tf.shape(x)
        shp4 = tf.concat(
            [shpC[:-2], (self.groups, self.filters)],
            axis=0,
        )
        x = tf.reshape(x, shp4)

        def swag():
            self.alpha_vector.assign(tf.broadcast_to(self.alpha, (self.groups,)))
            p = self.dirichlet.sample(self.sample_shape, seed=self.rng_seed)
            scaled = tf.expand_dims(p, -1) * x
            return tf.math.reduce_sum(scaled, axis=-2)

        def mean():
            return tf.math.reduce_mean(x, axis=-2)

        x = K.in_train_phase(swag, mean, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SwagTranspose(Swag):
    core_layer = kl.Conv2DTranspose


class SwagR(kl.Layer):
    """Build a SwagR residual layer
    This is closely related to ConvNeXt (see
    https://arxiv.org/pdf/2201.03545.pdf), except that
    a grouped convolution is performed on the last layer
    and then a resulting groups  are averaged with stochastic
    weights rather than using drop path. The averaging
    are sampled from a Dirichlet distribution, with parameters
    [alpha, ..., alpha].
    The motivation was to provides a gentler form of regularization
    than drop path. In practice, this converges more reliably in my
    application, but of of course YMMV. The amount of regularization
    can be controlled using `groups` and `alpha`: larger values for
    `groups` or `alpha` results in more regularization.
    We use *NormLayer* for normalization. In the ConvNextpaper, that was
    LayerNorm, but that does not train well in my application, so I typically
    use BatchNorm.
    The name derives from
            Stochastically weighted averaged groups – Residual
    Sure, it's slightly forced, but no worse than your average backronym.
    """

    NormLayer = kl.BatchNormalization
    rng_seed = (8, 88)

    def __init__(
        self,
        kernel_size: int = 7,
        scale: int = 4,
        groups: int = 8,
        alpha: Union[tf.Tensor, float] = 2.0,
    ):
        """Initialize SwagR layer
        Args:
            kernel_size (optional)
            scale (optional): Factor to increase inner filters by.
            groups (optional): Number of output groups (heads) to use
            alpha (optional): Controls how the groups are sampled during
                training. One corresponds to a uniform distribution, while
                larger values tend to sample primarily from one group
                at a time and smaller values tend to sample all groups
                simultaneously.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = scale
        self.groups = groups
        self.alpha = alpha

    def build(self, input_shape):
        n = input_shape[-1]
        assert n % self.groups == 0
        self.sample_shape = (input_shape[0], 1, 1)

        self.conv_lyr1 = kl.DepthwiseConv2D(
            kernel_size=self.kernel_size, padding="same"
        )
        self.norm_lyr = self.NormLayer(center=False, scale=False)
        self.conv_lyr2 = kl.Dense(self.scale * n, activation=gelu)
        self.rshp_lyr = kl.Reshape(
            input_shape[1:-1] + (self.groups, self.scale * n // self.groups)
        )
        self.conv_lyr3 = kl.Dense(n)
        self.alpha_vector = tf.Variable(
            [self.alpha] * self.groups, name="alpha", trainable=False, dtype="float32"
        )
        self.dirichlet = tfp.distributions.Dirichlet(self.alpha_vector)

    def call(self, x, training=None):
        skip = x
        x = self.conv_lyr1(x)
        x = self.norm_lyr(x)
        x = self.conv_lyr2(x)
        x = self.rshp_lyr(x)
        x = self.conv_lyr3(x)

        def swag():
            self.alpha_vector.assign(tf.broadcast_to(self.alpha, (self.groups,)))
            p = self.dirichlet.sample(self.sample_shape, seed=self.rng_seed)
            scaled = tf.expand_dims(p, -1) * x
            return tf.math.reduce_sum(scaled, axis=-2)

        def mean():
            return tf.math.reduce_mean(x, axis=-2)

        return skip + K.in_train_phase(swag, mean, training=training)
