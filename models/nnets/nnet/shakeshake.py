import keras
import keras.backend as K
from keras.layers import Layer


class ShakeShake2D(Layer):
    """ Shake-Shake-Image Layer
    
    See: https://arxiv.org/abs/1705.07485
    
    Typical usage:
    
    y = ShakeShake2D([build_branch(inputs),
                      build_branch(inputs)])

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = K.shape(x1)[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        beta = K.random_uniform((batch_size, 1, 1, 1))
        # shake-shake during training phase
        def x_shake():
            return (beta * x1 + (1 - beta) * x2 + 
                    K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2))
        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return K.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        s1, s2 = input_shape
        assert s1 == s2
        return s1