"""
Custom data augmentation functions.

"""
import numpy as np
from tensorflow.image import (flip_left_right, flip_up_down,  # central_crop,
                              resize, transpose)

rng = np.random.default_rng()


def shift_left_right(tile, pixels=1):
    if rng.integers(2):
        pixels *= -1  # 50% of the time left|right
    return np.roll(tile, pixels, 1)


def scale_and_crop(tile, scale_range=[1.05, 1.20]):
    """Zoom in, crop, and return scaling factor."""
    a, b = scale_range
    scale = (b - a) * rng.random() + a
    size = tile.shape[0]
    size_new = int(size * scale)
    scale = size_new / size  # effective scale
    tile_ = resize(tile, (size_new, size_new), method="lanczos3")
    # tile_ = resize(tile, (size_new, size_new), method="bilinear")
    tile_ = central_crop(tile_, size)
    if tile_.shape != tile.shape:
        print(f"[AUGMENT] shapes: {tile_.shape} {tile.shape}")
    assert tile_.shape == tile.shape
    return tile_, scale


def central_crop(tile, size):
    # `size` in pixels of cropped tile
    diff = tile.shape[0] - size
    p, q = int(np.ceil(diff / 2)), int(np.floor(diff / 2))
    return tile[p:-q, p:-q, :]
