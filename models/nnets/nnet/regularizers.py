def smooth_labels(y, smooth_factor=0.1):
    """Smooth a matrix of one-hot row-vector labels.

    y_smooth = (1 - α) * y_hot + α / K

    paper: https://tinyurl.com/2rb5mekv

    Args:
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    Returns:
        y_smooth: A matrix of smoothed labels.
    """
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        msg = f"Invalid label smoothing factor: {smooth_factor}"
        raise Exception(msg)
    return y


# Quick test
# import numpy as np  # noqa
# a = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f8")
# print(smooth_labels(a))
