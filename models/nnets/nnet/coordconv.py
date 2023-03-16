"""
From https://github.com/tchaton/CoordConv-keras

With minor edits to run on current Keras version
"""
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

from tensorflow.keras.layers import *

# TODO: find tf.keras version
from keras.utils import conv_utils


import keras.backend as K
def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

class CoordConv2D(Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 rank=2,
                 with_r=False,
                 **kwargs):
        super(CoordConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=rank + 2)
        self.rank=rank
        self.with_r = with_r
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        shift=2
        if self.with_r:
            shift+=1
        kernel_shape = self.kernel_size + (input_dim+shift, self.filters)
        #print(kernel_shape)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        
    
    def addCoord(self, inputs):
        
        shape = tf.shape(inputs)
        batch_size_tensor, x_dim, y_dim, c = shape[0], shape[1], shape[2], shape[3]
        
        xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.float32)
        
        xx_ones = tf.expand_dims(xx_ones, axis=-1)
        
        xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.cast(xx_range, tf.float32)
        xx_range = tf.expand_dims(xx_range, axis=1)
        
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, axis=-1)
              
        xx_channel = xx_channel / tf.cast(x_dim - 1, tf.float32)
        
        xx_channel = xx_channel*2 - 1
        
        ret = tf.concat([inputs, xx_channel, tf.transpose(xx_channel, (0, 2, 1, 3))], axis=-1)
        
        if self.with_r:
            
            rr = tf.sqrt( tf.square(xx_channel - .5) + tf.square(  tf.transpose(xx_channel) - .5))
            ret = tf.concat([ret, rr], axis=-1)
        return ret

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)
        
    def call(self, inputs):
        #Coordination
        
        inputs = self.addCoord(inputs)
        
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        else:
            raise Exception('rank should be two for now')
            
        if self.use_bias:
            outputs += self.bias
            
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(CoordConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def main():
    from keras.models import Model
    import numpy as np
    init = x = Input(shape=(64, 64, 3))
    x = CoordConv2D(32, (3, 3))(x)
    x = BatchNormalization()(x) 
    x  = CoordConv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = CoordConv2D(32, (3, 3))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    m = Model(init, x)
    m.compile('adam', 'mse')

    N = 20
    X = np.random.normal(0, 1, (N, 64, 64, 3))
    Y = np.random.randint(0, 2, N)
    m.fit(X, Y, epochs=100)

if __name__ == '__main__':
     main()
