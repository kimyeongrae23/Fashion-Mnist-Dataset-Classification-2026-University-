# coding: utf-8
"""
FashionConvNet: a NumPy/Scratch CNN for Fashion MNIST.

This model is based on the CNN source style from 'Deep Learning from Scratch':
Convolution, Relu, Pooling, Affine, SoftmaxWithLoss, and Adam optimizer are all
NumPy-based implementations. No deep-learning framework API is used.

Improvement over the textbook SimpleConvNet:
- The textbook basic structure is Conv-ReLU-Pool-Affine-ReLU-Affine-Softmax.
- Fashion MNIST is harder than digit MNIST, so this model uses two convolution
  blocks: Conv-ReLU-Pool -> Conv-ReLU-Pool -> Affine-ReLU-Affine-Softmax.
"""

import pickle
import numpy as np
from collections import OrderedDict

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
from common.gradient import numerical_gradient


class FashionConvNet:
    """CNN for Fashion MNIST using only NumPy-based textbook layers.

    Default architecture:
        input (N, 1, 28, 28)
        Conv1(1 -> 24, 3x3, pad=1) -> ReLU -> Pool(2x2)
        Conv2(24 -> 48, 3x3, pad=1) -> ReLU -> Pool(2x2)
        Affine(48*7*7 -> 256) -> ReLU -> Affine(256 -> 10)
        SoftmaxWithLoss
    """

    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv1_param=None,
                 conv2_param=None,
                 hidden_size=256,
                 output_size=10,
                 weight_init='he'):
        if conv1_param is None:
            conv1_param = {'filter_num': 24, 'filter_size': 3, 'pad': 1, 'stride': 1}
        if conv2_param is None:
            conv2_param = {'filter_num': 48, 'filter_size': 3, 'pad': 1, 'stride': 1}

        C, H, W = input_dim

        FN1 = conv1_param['filter_num']
        FS1 = conv1_param['filter_size']
        P1 = conv1_param['pad']
        S1 = conv1_param['stride']

        conv1_h = int((H + 2 * P1 - FS1) / S1 + 1)
        conv1_w = int((W + 2 * P1 - FS1) / S1 + 1)
        pool1_h = conv1_h // 2
        pool1_w = conv1_w // 2

        FN2 = conv2_param['filter_num']
        FS2 = conv2_param['filter_size']
        P2 = conv2_param['pad']
        S2 = conv2_param['stride']

        conv2_h = int((pool1_h + 2 * P2 - FS2) / S2 + 1)
        conv2_w = int((pool1_w + 2 * P2 - FS2) / S2 + 1)
        pool2_h = conv2_h // 2
        pool2_w = conv2_w // 2

        affine_input_size = FN2 * pool2_h * pool2_w

        def init_weight(shape, fan_in):
            if weight_init == 'he':
                scale = np.sqrt(2.0 / fan_in)
            elif weight_init == 'xavier':
                scale = np.sqrt(1.0 / fan_in)
            else:
                scale = float(weight_init)
            return scale * np.random.randn(*shape)

        self.params = {}
        self.params['W1'] = init_weight((FN1, C, FS1, FS1), C * FS1 * FS1)
        self.params['b1'] = np.zeros(FN1)

        self.params['W2'] = init_weight((FN2, FN1, FS2, FS2), FN1 * FS2 * FS2)
        self.params['b2'] = np.zeros(FN2)

        self.params['W3'] = init_weight((affine_input_size, hidden_size), affine_input_size)
        self.params['b3'] = np.zeros(hidden_size)

        self.params['W4'] = init_weight((hidden_size, output_size), hidden_size)
        self.params['b4'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], S1, P1)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], S2, P2)
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

        self.model_info = {
            'input_dim': input_dim,
            'conv1_output': (FN1, conv1_h, conv1_w),
            'pool1_output': (FN1, pool1_h, pool1_w),
            'conv2_output': (FN2, conv2_h, conv2_w),
            'pool2_output': (FN2, pool2_h, pool2_w),
            'affine_input_size': affine_input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
        }

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        total = x.shape[0]

        for i in range(0, total, batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / float(total)

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        for idx in (1, 2, 3, 4):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_params(self, file_name='params.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        layer_keys = ['Conv1', 'Conv2', 'Affine1', 'Affine2']
        for i, layer_key in enumerate(layer_keys, start=1):
            self.layers[layer_key].W = self.params['W' + str(i)]
            self.layers[layer_key].b = self.params['b' + str(i)]
