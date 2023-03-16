"""Generate batches of Keras training data

"""
import numpy as np
from tensorflow.keras.utils import Sequence

from ..nnet.transforms import (
    scale_and_crop,
    shift_left_right,
    flip_left_right,
    flip_up_down,
    transpose,
)

rng = np.random.default_rng()

transforms = [
    transpose,
    flip_up_down,
    flip_left_right,
    shift_left_right,
]

plot = False


class DataGenerator(Sequence):
    """Generate batched training data for Keras:

    Classification

    tile -> [presence]

    Parameters
    ----------
    indices : array of int or Zarr object
        Indices into the data contained in tiles and targets to generate
        data from. Typically corresponds to either the train or validation sets.
    tiles : namedtuple with arrays
        Single tuple with Array holding imagery data at 10 m resolution arranged
        as [example, height, width, channel]. This is a tuple to potentially support
        multiple inputs. (TODO: consider making an dict)
    targets : array of float
        Array holding the ground truth (target) values that we are trying to learn.
        Arranged as [target], where the target is *presence* (and *length* in future).
    batch_size : int
        Size of each bach.
    shuffle : bool, optional
        If true shuffle data at the end of each epoch. If False, data will be supplied
        in the order it appears in `indices`.
    augment : bool, optional
        Whether to apply augmentation. Set to False for validation.

    Notes
    -----
    tiles and targes are now passed as Zarr objects mapping the whole file:

        train_gen = DataGenerator(index.train, data)

        # where
        data.tiles
        data.presence
        data.length_m

    """

    def __init__(
        self,
        data,
        indices,
        batch_size=16,
        shuffle=True,
        augment=True,
        weight=False,
    ):
        self._rng = rng
        self._data = data
        self._base_indices = indices
        self.shuffle = shuffle  # before on_epoch_end!!!
        self.on_epoch_end()
        self.batch_size = batch_size
        self.height = data.tiles.shape[1]
        self.width = data.tiles.shape[2]
        self.channels = data.tiles.shape[3]
        self.samples = len(indices)
        self.augment = augment
        self.weight = weight

        assert hasattr(self._data, "tiles")
        assert hasattr(self._data, "presence")

    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self._indices) / self.batch_size))

    def __getitem__(self, batch_index):
        """Load one batch of data into memory.

        Parameters
        ==========
        batch_index : int
            Index of data batch to generate:
            batch_0, batch_1, .., batch_n

        Returns
        =======
        X : np.array (4D)
            Block of data (dual-band thumbnails) with shape
            `[batch_size, height, width, channels]`
        Y : np.array (1D)
            Target value for the model, with shape `[batch_size]`.
        """
        indices = self._indices[
            batch_index * self.batch_size: (batch_index + 1) * self.batch_size
        ]
        return self._data_generation(indices)  # loads one batch in memory

    def on_epoch_end(self):
        """Updates _indices after each epoch"""
        # self._indices = self._base_indices.copy()
        self._indices = self._base_indices[:]  # for zarr
        if self.shuffle:
            self._rng.shuffle(self._indices)

    def _data_generation(self, indices):
        """Generate one batch of (augmented) data at indices.

        Parameters
        ----------
        indices : sequence of int
            These are indices into tiles and targets.

        Returns
        -------
        X : 4D array of floats
            Image inputs for the model (tiles).
        y : 1D or 2D array of ints|floats
            Target values for the model (labels).
        """
        X = np.empty((self.batch_size, self.height, self.width, self.channels))
        y = np.empty((self.batch_size), dtype=int)

        # Avoid loading full array into memory
        for i, idx in enumerate(indices):
            tile = self._data.tiles[idx]
            target = self._data.presence[idx]

            # Apply data augmentation 50% of the time
            if self.augment:
                for aug in transforms:
                    if self._rng.integers(2):
                        tile = aug(tile)

            X[i] = tile
            y[i] = target

        return (X,), y[:, None]


class DataGenerator2(DataGenerator):
    """Generate batched training data for Keras (for data v1):

    From detection_tiles_v1.zarr
    Classification + Regression with sample weights.
    Extended to include length and weights.

    Fields in data v1:

        - tiles
        - lon
        - lat
        - detect_id
        - length
        - weight

    tile -> weigth * [presence, lenght]

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self._data, "length")

    def _data_generation(self, indices):
        """Generate one batch of (augmented) data at indices.

        Parameters
        ----------
        indices : sequence of int
            These are indices into tiles and targets.

        Returns
        -------
        X : 4D array of floats
            Image inputs for the model (tiles).
        y : dict with 1D arrays of floats
            Target values for the model (labels).
        weight: 1D array of floats
            Weight for each (length) sample.

        """
        X = np.empty((self.batch_size, self.height, self.width, self.channels))
        presence = np.empty((self.batch_size, 1))
        length = np.empty((self.batch_size, 1))
        weight = np.ones((self.batch_size, 1))

        # Avoid loading full array into memory
        for i, idx in enumerate(indices):
            tile = self._data.tiles[idx]
            length_ = self._data.length[idx]  # array?

            if plot:
                tile_orig = tile.copy()
                length_orig = length_.copy()
                scale = 1

            # Apply data augmentation 50% of the time
            if self.augment:

                for aug in transforms:
                    if self._rng.integers(2):
                        tile = aug(tile)

                if self._rng.integers(2):
                    tile, scale = scale_and_crop(tile)
                    length_ *= scale

            X[i] = tile
            length[i] = length_

            presence[i] = self._data.presence[idx]

            if self.weight and hasattr(self._data, "weight"):
                weight[i] = self._data.weight[idx]

            if plot:
                import matplotlib.pyplot as plt

                plt.matshow(tile_orig[:, :, 0], cmap="bone")
                plt.title(f"length={length_orig:.1f}")
                plt.matshow(X[i, :, :, 0], cmap="bone")
                plt.title(f"length={length_:.1f} (scale={scale:.2f})")
                plt.show()

        return X, {"presence": presence, "length": length}, weight


class DataGenerator3(DataGenerator):
    """Generate batched training data for Keras (for data v2/v3):

    From detection_tiles_v2.zarr
    For Classification + Regression with sample weights.
    Extended to include length and weights.
    Extended for fancy indexing on cloud Zarr.
    Pulls full batch of data into local memory.

    Fields in data v2/v3:

        - tiles
        - detect_lon
        - detect_lat
        - detect_id
        - length_m
        - weight

    tile -> weigth * [presence, lenght_m]

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self._data, "length_m")

    def _pull_batch(self, indices):
        """Pull a batch of data from GCS to local memory.

        Define (x, y, z) indices for fancy indexing.
        Do not use range()!
        Cast types float16 -> float32.

        """
        ii = np.arange(self.height)  # 80x80
        jj = np.arange(self.width)
        kk = np.arange(self.channels)

        tiles = self._data.tiles.oindex[(indices, ii, jj, zz)].astype('f4')
        presence = self._data.presence.vindex[indices].astype('f4')
        length = self._data.length_m.vindex[indices].astype('f4')

        if self.weight and hasattr(self._data, "weight"):
            weight = self._data.weight.vindex[indices].astype('f4')
        else:
            weight = np.ones_like(presence)

        return tiles, presence, length, weight

    def _data_generation(self, indices):
        """Generate one batch of (augmented) data at indices.

        Parameters
        ----------
        indices : sequence of int
            These are indices into tiles and targets.

        Returns
        -------
        X : 4D array of floats
            Image inputs for the model (tiles).
        y : dict with 1D arrays of floats
            Target values for the model (labels).
        weight: 1D array of floats
            Weight for each (length) sample.

        """
        tiles, presence, length, weight = self._pull_batch(indices)

        # Perform data agumentation on batch
        for i in range(self.batch_size):

            tile_ = tiles[i]
            length_ = np.array(length[i])

            if plot:
                tile_orig = tile_.copy()
                length_orig = length_.copy()
                scale = 1

            # Apply augmentation 50% of the time
            if self.augment:

                for aug in transforms:
                    if self._rng.integers(2):
                        tile_ = aug(tile_)

                if self._rng.integers(2):
                    tile_, scale = scale_and_crop(tile_)
                    length_ *= scale

            tiles[i] = tile_
            length[i] = length_

            if plot:
                import matplotlib.pyplot as plt

                plt.matshow(tile_orig[:, :, 0], cmap="bone")
                plt.title(f"length={length_orig:.1f}")
                plt.matshow(tiles[i, :, :, 0], cmap="bone")
                plt.title(f"length={length_:.1f} (scale={scale:.2f})")
                plt.show()

        return tiles, {"presence": presence, "length": length}, weight
