# coding: utf-8
"""
Basic functions from the style of 'Deep Learning from Scratch'.
This file intentionally uses NumPy only. Do not import PyTorch/TensorFlow/Keras here.
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """Numerically stable softmax. Supports 1D and 2D inputs."""
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x)
        return y / np.sum(y, axis=1, keepdims=True)

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    """
    Cross entropy loss.
    y: predicted probability, shape (N, C) or (C,)
    t: label index shape (N,) or one-hot shape (N, C)
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # one-hot label -> label index
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
