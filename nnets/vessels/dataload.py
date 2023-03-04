"""
Generate batches of Keras data (tiles for inference)

"""
import numpy as np
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    """Load batched tiles for Keras:

    Parameters
    ----------
    tiles : out of memory zarr 3d array
    indices : in memory numpy 1d array
    batch_size : int (default 16 tiles)

    Example
    -------
    data_load = DataLoader(tiles, index, batch_size=16)
    model.predict(data_load)

    """

    def __init__(
        self,
        tiles,
        indices,
        batch_size=16,
    ):
        self._tiles = tiles
        self._indices = indices
        self.batch_size = batch_size
        self.tile_size = self._tiles.shape[1]
        self.samples = len(indices)

    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self._indices) / self.batch_size))

    def __getitem__(self, batch_index):
        """Load one batch of data into memory.

        Parameters
        ==========
        batch_index : int
            Index of data batch to generate.
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
        return self._pull_batch(indices)

    # def on_epoch_end(self):
    #     """Updates _indices after each epoch"""
    #     # self._indices = self._base_indices.copy()
    #     self._indices = self._base_indices[:]  # for zarr
    #     if self.shuffle:
    #         self._rng.shuffle(self._indices)

    def _pull_batch(self, indices):
        """Pull a batch of data from GCS to local memory.

        Define (ix, iy, iz) indices for fancy indexing.
        Do not use range(), zarr doesn't like it.
        Cast types float16 -> float32.

        """
        iz = np.arange(2)                    # channels
        ix = iy = np.arange(self.tile_size)  # 80x80
        return self._tiles.oindex[(indices, ix, iy, iz)].astype('f4')
