from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import BatchNormalization
from keras.utils import get_custom_objects
from keras.layers import BatchNormalization, Reshape


def gbnorm(y, h, **kwargs):
    n0 = {'Z2' : 1, 'C4': 4, 'D4' : 8}[h]
    # Assumes 'D4'
    n = y.shape[-1]
    p = y.shape[1:-1]
    y = Reshape(p + (n // n0, n0))(y)
    y = BatchNormalization(axis=-2, **kwargs)(y)
    y = Reshape(p + (n,))(y)
    return y

def gscalarpool(y, h):
    n0 = {'Z2' : 1, 'C4': 4, 'D4' : 8}[h]
    # Assumes 'D4'
    n = y.shape[-1]
    p = y.shape[1:-1]
    y = Reshape(p + (n // n0, n0))(y)
    return K.mean(y, axis=-1)

class GBatchNorm(BatchNormalization):

    def __init__(self, h, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        self.h = h
        if axis != -1:
            raise ValueError('Assumes 2D input with channels as last dimension.')
        super(GBatchNorm, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                         moving_mean_initializer=moving_mean_initializer,
                                         moving_variance_initializer=moving_variance_initializer,
                                         beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                                         beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        self.gconv_indices, self.gconv_shape_info, w_shape = gconv2d_util(h_input=self.h, h_output=self.h,
                                                                          in_channels=input_shape[-1],
                                                                          out_channels=input_shape[-1],
                                                                          ksize=1)
        if self.h == 'C4':
            dim //= 4
        elif self.h == 'D4':
            dim //= 8
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        def repeat(w):
            n = 1
            if self.h == 'C4':
                n *= 4
            elif self.h == 'D4':
                n *= 8
            elif self.h == 'Z2':
                n *= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)

            return K.reshape(
                K.tile(
                    K.expand_dims(w, -1), [1, n]), [-1])

        self.repeated_gamma = repeat(self.gamma) if self.scale else None
        self.repeated_beta = repeat(self.beta) if self.center else None

        self.repeated_moving_mean = repeat(self.moving_mean)
        self.repeated_moving_variance = repeat(self.moving_variance)
        self.built = True

    def call(self, inputs, training=None):

        def unrepeat(w):
            n = 1
            if self.h == 'C4':
                n *= 4
            elif self.h == 'D4':
                n *= 8
            elif self.h == 'Z2':
                n *= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)

            return K.mean(
                K.reshape(w, (K.int_shape(w)[0] // n, n)), -1)

        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.repeated_moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.repeated_moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.repeated_beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.repeated_gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.repeated_moving_mean,
                    self.repeated_moving_variance,
                    self.repeated_beta,
                    self.repeated_gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.repeated_gamma, self.repeated_beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 unrepeat(mean),
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 unrepeat(variance),
                                                 self.momentum)])

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        return dict(list({'h': self.h}.items()) +
                    list(super(GBatchNorm, self).get_config().items()))


get_custom_objects().update({'GBatchNorm': GBatchNorm})





import keras.backend as K
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util
from keras.engine import InputSpec
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.utils import get_custom_objects
from keras_gcnn.transform_filter import transform_filter_2d_nhwc

scale_map = {'C4' : 4, 'D4' : 8, 'Z2' : 1}


class GConv2D(Conv2D):
    def __init__(self, filters, kernel_size, h_input, h_output, strides=(1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, transpose=False, **kwargs):
        """
        :param filters:
        :param kernel_size:
        :param h_input:
        :param h_output:
        :param h_input: one of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        :param h_output: one of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
              The choice of h_output of one layer should equal h_input of the next layer.
        :param strides:
        :param padding:
        :param data_format:
        :param dilation_rate:
        :param activation:
        :param use_bias:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """
        if use_bias:
            raise NotImplementedError('Does not support bias yet')  # TODO: support bias

        if not isinstance(kernel_size, int) and not kernel_size[0] == kernel_size[1]:
            raise ValueError('Requires square kernel')

        self.h_input = h_input
        self.h_output = h_output
        self.transpose = transpose

        super(GConv2D, self).__init__(filters, kernel_size, strides=strides, padding=padding, data_format=data_format,
                                      dilation_rate=dilation_rate, activation=activation,
                                      use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint, **kwargs)

    def compute_output_shape(self, input_shape):
        if self.transpose:
            shape = Conv2DTranspose.compute_output_shape(self, input_shape)
        else:
            shape = super(GConv2D, self).compute_output_shape(input_shape)
        nto = shape[3]

        if self.h_output == 'C4':
            nto *= 4
        elif self.h_output == 'D4':
            nto *= 8
        return (shape[0], shape[1], shape[2], nto)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            raise NotImplementedError('Channels first is not implemented for GConvs yet.')
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        orig_input_dim = input_dim
        input_dim //= scale_map[self.h_input]

        self.gconv_indices, self.gconv_shape_info, w_shape = gconv2d_util(h_input=self.h_input, h_output=self.h_output,
                                                                          in_channels=input_dim,
                                                                          out_channels=self.filters,
                                                                          ksize=self.kernel_size[0])

        self.kernel = self.add_weight(shape=w_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
              name='bias',
              shape=(self.filters // scale_map[self.h_output]),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              trainable=True,
              dtype=self.dtype)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: orig_input_dim})
        self.built = True

    def call(self, inputs):
        outputs = gconv2d(
            inputs,
            self.kernel,
            self.gconv_indices,
            self.gconv_shape_info,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            transpose=self.transpose,
            output_shape=self.compute_output_shape(inputs.shape))

        if self.bias is not None:
            scale = {'C4' : 4, 'D4' : 8, 'Z2' : 1}[self.h_input]
            n = outputs.shape[-1]
            p = outputs.shape[1:-1]
            n0 = scale_map[self.h_output]
            outputs = K.reshape(outputs, (-1,) + p + (n // n0, n0))
            b = K.reshape(self.bias, (1, 1, 1, n // n0, 1))
            outputs = K.add(outputs, b)
            outputs = K.reshape(outputs, (-1,) + p + n)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(GConv2D, self).get_config()
        config['h_input'] = self.h_input
        config['h_output'] = self.h_output
        return config


def gconv2d(x, kernel, gconv_indices, gconv_shape_info, strides=(1, 1), padding='valid',
            data_format=None, dilation_rate=(1, 1), transpose=False, output_shape=None):
    """2D group equivariant convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow data format
            for inputs/kernels/ouputs.
        dilation_rate: tuple of 2 integers.
    # Returns
        A tensor, result of 2D convolution.
    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    # Transform the filters
    transformed_filter = transform_filter_2d_nhwc(w=kernel, flat_indices=gconv_indices, shape_info=gconv_shape_info)
    if transpose:
        output_shape = (K.shape(x)[0], output_shape[1], output_shape[2], output_shape[3])
        transformed_filter = transform_filter_2d_nhwc(w=kernel, flat_indices=gconv_indices, shape_info=gconv_shape_info)
        transformed_filter = K.permute_dimensions(transformed_filter, [0, 1, 3, 2])
        return K.conv2d_transpose(x=x, kernel=transformed_filter, output_shape=output_shape, strides=strides,
                                padding=padding, data_format=data_format)
    return K.conv2d(x=x, kernel=transformed_filter, strides=strides, padding=padding, data_format=data_format,
                    dilation_rate=dilation_rate)


get_custom_objects().update({'GConv2D': GConv2D})