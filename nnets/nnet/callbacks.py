from tensorflow.keras.callbacks import Callback


class LinearDecay(Callback):
    def __init__(self, var, epochs, init_value, final_value, first_epoch=0):
        self.var = var
        self.epochs = max(epochs // 4, 1)
        self.init_value = init_value
        self.final_value = final_value
        self.first_epoch = first_epoch

    def on_epoch_begin(self, epoch, logs=None):
        epoch = max(min(self.epochs, epoch) - self.first_epoch, 0)
        delta = self.final_value - self.init_value
        rng = self.epochs - self.first_epoch
        self.var.assign(self.init_value + delta * epoch / rng)
