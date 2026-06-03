# coding: utf-8
"""
Fashion MNIST loader for the Scratch/NumPy CNN implementation.

This loader reads the four original .gz IDX files downloaded from:
https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

No PyTorch/TensorFlow/Keras is used in this file.
"""

import os
import gzip
import struct
import numpy as np


_FILE_NAMES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
}


def _one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


def _load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid image file: {path}, magic={magic}')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, 1, rows, cols)
    return data


def _load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid label file: {path}, magic={magic}')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_fashion_mnist(base_dir='.', normalize=True, flatten=False, one_hot_label=True):
    """
    Load Fashion MNIST from local .gz files.

    Parameters
    ----------
    base_dir : str
        Directory containing the four .gz files.
    normalize : bool
        If True, image values are converted to float32 and scaled to [0, 1].
    flatten : bool
        If True, images are reshaped to (N, 784). For CNN, use False.
    one_hot_label : bool
        If True, labels are returned as one-hot vectors.

    Returns
    -------
    (x_train, t_train), (x_test, t_test)
    """
    paths = {key: os.path.join(base_dir, name) for key, name in _FILE_NAMES.items()}

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Missing {key}: {path}\n'
                'Copy the four Fashion MNIST .gz files into the Scratch folder.'
            )

    x_train = _load_images(paths['train_images'])
    t_train = _load_labels(paths['train_labels'])
    x_test = _load_images(paths['test_images'])
    t_test = _load_labels(paths['test_labels'])

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    else:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    if one_hot_label:
        t_train = _one_hot(t_train, 10)
        t_test = _one_hot(t_test, 10)

    return (x_train, t_train), (x_test, t_test)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_fashion_mnist()
    print('x_train:', x_train.shape, x_train.dtype)
    print('t_train:', t_train.shape, t_train.dtype)
    print('x_test :', x_test.shape, x_test.dtype)
    print('t_test :', t_test.shape, t_test.dtype)
