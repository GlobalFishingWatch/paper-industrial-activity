"""Generate batches of Keras training data.

Pull random batches from GCS Zarr.
Apply data augmentations.
Format data for multi and single inputs.

Augmentations:
    # Gaussian Noise
    Horizontal & Vertical flips
    Small affine rotations (up to 5 degrees)
    Small scale change (+/- 10% image size)
    # Random brightness & contrast change
    # Minor Elastic transformation

"""
import numpy as np

import albumentations as A
import tensorflow as tf

from ..nnet.regularizers import smooth_labels

CLASSES = ["wind", "oil", "other", "noise"]

rng = np.random.default_rng(123)

# Augmentation pipeline (p=0.5 each)
transform = A.Compose([  # noqa
    A.Flip(),
    A.Transpose(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=5),
    # A.RandomBrightnessContrast(),
    # A.ElasticTransform(),
], p=0.5)


class DatasetSource:
    """Generate batched training data for Keras

    Multi input (model_multi)

    """

    num_classes = len(CLASSES)
    # base_size1 = 200
    base_size1 = 100
    base_size2 = 100
    channels1 = 2
    channels2 = 4

    def __init__(
        self,
        data,
        indices=None,
        shuffle=True,
        augment=False,
        smooth=0,
        batch_size=32,
    ):
        assert hasattr(data, "tiles_s1")
        assert hasattr(data, "tiles_s2")
        assert hasattr(data, "label")

        if indices is None:
            indices = np.arange(len(data))

        self._rng = rng
        self._data = data
        self.indices = indices
        self.shuffle = shuffle
        self.augment = augment
        self.smooth = smooth
        self.batch_size = batch_size

    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self.indices) / self.batch_size))

    @property
    def shape_x1(self):
        return (self.base_size1, self.base_size1, self.channels1)

    @property
    def shape_x2(self):
        return (self.base_size2, self.base_size2, self.channels2)

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
        # TODO: time downloading one at a time (i.e. previous approach)
        indices = indices.numpy()  # important !!!
        ii1 = jj1 = np.arange(self.base_size1)  # 200
        ii2 = jj2 = np.arange(self.base_size2)  # 100
        kk1 = np.arange(self.channels1)  # 2
        kk2 = np.arange(self.channels2)  # 4

        X1 = self._data.tiles_s1.oindex[(indices, ii1, jj1, kk1)]
        X2 = self._data.tiles_s2.oindex[(indices, ii2, jj2, kk2)]
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

        Y = one_hot(Y)  # str -> floats

        if self.smooth > 0:
            Y = smooth_labels(Y, self.smooth)  # batch LabelSmoothing

        if self.augment:
            for i in range(self.batch_size):
                img1, img2 = X1[i], X2[i]
                img1 = transform(image=img1)["image"]
                img2 = transform(image=img2)["image"]
                X1[i], X2[i] = img1, img2

        return X1, X2, Y


class DatasetSource2(DatasetSource):
    """Single input (model_single)."""

    tile_size = 400
    channels = 6

    def dataset(self):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        def batch_indices_to_data(indices):
            X, Y = tf.py_function(
                func=self.batch_indices_to_data,
                inp=[indices],
                Tout=(tf.float32, tf.float32),
            )
            shape_x = (self.tile_size, self.tile_size, self.channels)
            X = tf.ensure_shape(X, (self.batch_size, *shape_x))
            Y = tf.ensure_shape(Y, (self.batch_size, *self.shape_y))
            return X, Y  # <-- input to the Model

        return (
            self.base_dataset()
            .batch(self.batch_size, drop_remainder=True)
            .map(batch_indices_to_data, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(1)
            .with_options(self.get_ds_options())
        )

    def batch_indices_to_data(self, indices):
        """Generate one batch of data

        Parameters
        ==========
        index : sequence of int

        Returns
        =======
        X : model inputs
        Y : model outputs

        """
        X1, X2, Y = self.pull_batch_from_zarr(indices)

        Y = one_hot(Y)  # str -> floats

        # Make both tiles equal size and stack
        size = self.tile_size
        pads = tf.constant([[0, 0], [150, 150], [150, 150], [0, 0]])
        X1 = resize(X1, (size, size), method="lanczos3")
        X2 = tf.pad(X2, pads, "CONSTANT")
        X = np.zeros((self.batch_size, size, size, self.channels))

        for i in range(self.batch_size):
            X[i] = np.dstack([X1[i], X2[i]])

        # Perform data agumentation on batch
        for i in range(self.batch_size):

            img = X[i]

            # Apply augmentation 50% of the time
            if self.augment:

                for aug in augmentations:
                    if self._rng.integers(2):
                        img = aug(img)

                if self._rng.integers(2):
                    img, scale = scale_and_crop(img)

                if self._rng.integers(2):
                    img[:, :, :, 2:] = np.zeros_like(img[:, :, :, 2:])  # only for S2

            X[i] = img

        return X, Y


class DatasetSource3(DatasetSource2):
    """Smaller tile size single input (model_single2)."""

    tile_size = 200

    def batch_indices_to_data(self, indices):
        """Generate one batch of data

        Args:
            index (list[int]): sequence of int

        Returns:
            X: model inputs
            Y: model outputs

        """
        X1, X2, Y = self.pull_batch_from_zarr(indices)

        Y = one_hot(Y)  # str -> floats

        # Make both tiles equal size and stack
        X1 = resize(X1, (400, 400), method="lanczos3")
        X1 = X1[:, 100:300, 100:300, :]  # crop half the size

        pads = tf.constant([[0, 0], [50, 50], [50, 50], [0, 0]])
        X2 = tf.pad(X2, pads, "CONSTANT")

        X = np.zeros((self.batch_size, 200, 200, self.channels))

        for i in range(self.batch_size):
            X[i] = np.dstack([X1[i], X2[i]])

        # Perform data agumentation on batch
        for i in range(self.batch_size):

            img = X[i]

            # Apply augmentation 50% of the time
            if self.augment:

                for aug in augmentations:
                    if self._rng.integers(2):
                        img = aug(img)

                """
                if self._rng.integers(2):
                    img1, scale = scale_and_crop(img1)

                if self._rng.integers(2):
                    img2 = np.zeros_like(img2)  # only for S2
                """

            X[i] = img

        return X, Y


class DatasetSource4(DatasetSource):
    """Smaller tile size multi input (model_multi2)."""

    base_size1 = 100  # half the original size
    base_size2 = 100

    def pull_batch_from_zarr(self, indices):
        # TODO: time downloading one at a time (i.e. previous approach)
        indices = indices.numpy()  # important !!!
        ii1 = jj1 = np.arange(self.base_size1 * 2)  # 200
        ii2 = jj2 = np.arange(self.base_size2)  # 100
        kk1 = np.arange(self.channels1)  # 2
        kk2 = np.arange(self.channels2)  # 4

        X1 = self._data.tiles_s1.oindex[(indices, ii1, jj1, kk1)]
        X2 = self._data.tiles_s2.oindex[(indices, ii2, jj2, kk2)]
        Y = self._data.label.vindex[indices]

        X1 = X1[:, 50:150, 50:150, :]  # crop half the size

        return X1.astype("f4"), X2.astype("f4"), Y


def one_hot(labels, classes=CLASSES):
    """One-hot encode batch of labels -> tensor."""
    indices = [classes.index(label) for label in labels]
    return tf.one_hot(indices, len(classes))


# instead of maxing lambda_t out at 1, max it out at 0.1
