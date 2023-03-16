"""Generate batches of Keras training data.

Pull random batches from GCS Zarr.
Format data for multi-input mixed-data net.
Apply data normalization [1].
Apply data augmentations.

[1] order-magnitude normalization:
https://tinyurl.com/49yxhh8j

"""
import numpy as np

import albumentations as A
import tensorflow as tf

from ..nnet.regularizers import smooth_labels

CLASSES = ["fishing", "nonfishing"]

BASE_SIZE = 100
TILE_SIZE = 100
NUM_CHANNELS = 11

LENGTH_DIVIDER = 100

CHANNEL_DIVIDERS = np.array([
    100,
    1,
    10000,
    1000000,
    10,
    10,
    10,
    1,
    0.1,
    0.1,
    10,
])

rng = np.random.default_rng(12345)

# Augmentation pipeline
transform = A.Compose(
    [  # noqa
        A.Flip(p=1),
        A.Transpose(p=0.75),
        # A.RandomRotate90(p=0.75),
        A.ChannelDropout(fill_value=0.0, p=0.5),
        A.CoarseDropout(max_holes=15, max_height=5, max_width=5, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=360, p=0.75),
        # A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
    ],
    p=0.5,
)


def random_scale(x, a=0.9, b=1.1):
    """Randomly scale x by a factor between a and b."""
    if rng.integers(2):
        x *= (b - a) * rng.random() + a
    return x


def transform_channels(img, channels=[4, 5]):
    """Log-transform selected channels."""
    for c in channels:
        img[..., c] = np.log(1 + img[..., c])
    return img


def normalize_channels(img, dividers=CHANNEL_DIVIDERS):
    return img / np.array(dividers, ndmin=4, dtype="f4")


def normalize_array(arr, divider=LENGTH_DIVIDER):
    return arr / float(divider)


def blank_channels(img, channels=[6, 7, 8, 9, 10]):
    blank = [0 if c in channels else 1 for c in range(img.shape[-1])]
    return img * np.array(blank, ndmin=4, dtype="f4")


def crop_tiles(img, size):
    orig_size = img.shape[1]
    if orig_size > size:
        px = int((orig_size - size) / 2.0)
        img = img[:, px:-px, px:-px, :]
    return img


def one_hot(labels, classes=CLASSES):
    """One-hot encode batch of labels -> tensor."""
    indices = [classes.index(label) for label in labels]
    return tf.one_hot(indices, len(classes))


class DatasetSource:
    """Generate batched training data for Keras."""

    def __init__(
        self,
        data,
        indices=None,
        shuffle=True,
        augment=False,
        normalize=True,
        smooth=0,
        batch_size=64,
        num_classes=len(CLASSES),
        base_size=BASE_SIZE,
        tile_size=TILE_SIZE,
        num_channels=NUM_CHANNELS,
    ):
        assert hasattr(data, "tiles")
        assert hasattr(data, "length_m")
        assert hasattr(data, "label")

        if indices is None:
            indices = np.arange(len(data))

        self._rng = rng
        self._data = data
        self.indices = indices
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.smooth = smooth
        self.batch_size = batch_size
        self.base_size = base_size
        self.tile_size = tile_size
        self.num_channels = num_channels
        self.num_classes = num_classes

    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self.indices) / self.batch_size))

    @property
    def shape_x1(self):
        return (self.tile_size, self.tile_size, self.num_channels)

    @property
    def shape_x2(self):
        return (1,)  # scalar value

    @property
    def shape_y(self):
        return (self.num_classes,)

    def get_ds_options(self):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        return options

    def base_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self.indices)
        if self.shuffle:
            ds = ds.shuffle(len(self.indices))
        return ds.with_options(self.get_ds_options())

    def dataset(self):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        def batch_indices_to_data(indices):
            X1, X2, Y = tf.py_function(
                func=self.batch_indices_to_data,
                inp=[indices],
                Tout=(tf.float32, tf.float32, tf.float32),
            )
            X1 = tf.ensure_shape(X1, (self.batch_size, *self.shape_x1))
            X2 = tf.ensure_shape(X2, (self.batch_size, *self.shape_x2))
            Y = tf.ensure_shape(Y, (self.batch_size, *self.shape_y))
            return (X1, X2), Y  # <-- NOTE: input to the Model

        return (
            self.base_dataset()
            .batch(self.batch_size, drop_remainder=True)
            .map(batch_indices_to_data, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(1)
            .with_options(self.get_ds_options())
        )

    def pull_batch_from_zarr(self, indices):
        indices = indices.numpy()  # important !!!
        ii = jj = np.arange(self.base_size)
        kk = np.arange(self.num_channels)
        X1 = self._data.tiles.oindex[(indices, ii, jj, kk)]
        X2 = self._data.length_m.vindex[indices]
        Y = self._data.label.vindex[indices]
        return X1.astype("f4"), X2.astype("f4"), Y

    def batch_indices_to_data(self, indices):
        """Generate one batch of (augmented) data.

        Args:
            index: sequence of int

        Returns:
            X: model inputs, X = (X1, X2)
            Y: model outputs, class scores
        """
        X1, X2, Y = self.pull_batch_from_zarr(indices)

        X1 = crop_tiles(X1, self.tile_size)

        X2 = X2[:, np.newaxis]  # shape -> (batch, 1)

        Y = one_hot(Y)  # str -> floats

        if self.smooth > 0:
            Y = smooth_labels(Y, self.smooth)  # batch LabelSmoothing

        if self.normalize:
            X1 = transform_channels(X1)
            X1 = normalize_channels(X1)
            X2 = normalize_array(X2)

        if self.augment:
            for i in range(self.batch_size):
                img, length = X1[i], X2[i]
                img = transform(image=img)["image"]
                length = random_scale(length)
                X1[i], X2[i] = img, length

        return X1, X2, Y
