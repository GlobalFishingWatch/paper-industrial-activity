from tensorflow import keras
import tensorflow.keras.backend as K


class DropBlock2D(keras.layers.Layer):
    """DropBlock Regularization layer
    
    See: https://arxiv.org/pdf/1810.12890.pdf

    Originally from https://github.com/CyberZHG/keras-drop-block ,
    But may have some modifications to support unspecified
    grid sizes.
    """

    def __init__(self,
                 block_size,
                 keep_prob,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.data_format = data_format
        self.supports_masking = True

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_gamma(self, shape):
        """Get the number of activation units to drop"""
        height, width = K.cast(shape[1], K.floatx()), K.cast(shape[2], K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))


    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma(shape))
        half_block_size = self.block_size // 2
        mask = mask[:, half_block_size:-half_block_size, half_block_size:-half_block_size] 
        mask = K.spatial_2d_padding(mask, ((half_block_size, half_block_size), 
                                              (half_block_size, half_block_size)))
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            mask = self._compute_drop_mask(shape)
            outputs = (outputs * mask  / (K.mean(mask, axis=(1, 2, 3), keepdims=True) + 1e-9))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)