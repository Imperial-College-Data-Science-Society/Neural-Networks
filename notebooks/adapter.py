# scientific computing library
import numpy as np


def make_spiral(N, dims, num_classes, one_hot=False):
    """Generate toy spiral data for multiclass classification."""
    X = np.zeros(shape=(N * num_classes, dims))
    y = np.zeros(shape=(N * num_classes), dtype=int)
    for j in range(num_classes):
        radius = np.linspace(0.0, 1, N)
        angle = np.linspace(j * 4, (j + 1) * 4, N)
        ix = range(N * j, N * (j + 1))
        X[ix] = np.c_[radius * np.sin(angle), radius * np.cos(angle)]
        y[ix] = j

    if one_hot:
        return X, array_to_onehot(y, num_classes)
    else:
        return X, y


def int_to_onehot(x, num_classes):
    """Convert an `int` to one hot (binary) format."""
    tmp = np.zeros(num_classes, dtype=int)
    tmp[-x - 1] = 1
    return tmp


def array_to_onehot(arr, num_classes):
    """Convert an `np.array(dtype=int)` to one hot (binary) format."""
    tmp = np.zeros(shape=(arr.shape[0], num_classes), dtype=int)
    for j in range(len(arr)):
        tmp[j] = int_to_onehot(arr[j], num_classes)
    return tmp
