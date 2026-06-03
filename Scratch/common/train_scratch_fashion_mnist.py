# coding: utf-8
"""
Train FashionConvNet on Fashion MNIST using only NumPy/Scratch code.

Important assignment condition:
- This script does NOT use PyTorch, TensorFlow, Keras, torchvision, or autograd.
- CNN layers are implemented with the textbook-style NumPy classes in common/.

Required files in the same Scratch folder:
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz
"""

import os
import argparse
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from fashion_mnist_loader import load_fashion_mnist
from fashion_convnet import FashionConvNet
from common.optimizer import Adam, SGD


FORBIDDEN_FRAMEWORK_NAMES = ('torch', 'torchvision', 'tensorflow', 'keras')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def make_optimizer(name, lr):
    name = name.lower()
    if name == 'adam':
        return Adam(lr=lr)
    if name == 'sgd':
        return SGD(lr=lr)
    raise ValueError('optimizer must be adam or sgd')


def save_graphs(train_acc_list, test_acc_list, train_loss_list, output_dir='.'):
    epochs = np.arange(1, len(train_acc_list) + 1)

    plt.figure()
    plt.plot(epochs, train_acc_list, label='train acc')
    plt.plot(epochs, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.0)
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'accuracy_graph.png'), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, len(train_loss_list) + 1), train_loss_list, label='train loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'loss_graph.png'), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--seed', type=int, default=42)

    # Model size. Default is intentionally stronger than textbook SimpleConvNet,
    # but it still uses only textbook-style NumPy layers.
    parser.add_argument('--conv1_filters', type=int, default=24)
    parser.add_argument('--conv2_filters', type=int, default=48)
    parser.add_argument('--hidden_size', type=int, default=256)

    # Useful for quick debugging. For final submission, keep both as 0.
    parser.add_argument('--max_train', type=int, default=0)
    parser.add_argument('--max_test', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    print('Scratch Fashion MNIST CNN training')
    print('Framework check: PyTorch/TensorFlow/Keras are not imported.')
    print('Used modules: numpy, matplotlib, gzip/struct loader, textbook-style common layers')

    (x_train, t_train), (x_test, t_test) = load_fashion_mnist(
        base_dir=args.data_dir,
        normalize=True,
        flatten=False,
        one_hot_label=True
    )

    if args.max_train and args.max_train > 0:
        x_train = x_train[:args.max_train]
        t_train = t_train[:args.max_train]
    if args.max_test and args.max_test > 0:
        x_test = x_test[:args.max_test]
        t_test = t_test[:args.max_test]

    print('x_train:', x_train.shape, 't_train:', t_train.shape)
    print('x_test :', x_test.shape, 't_test :', t_test.shape)

    network = FashionConvNet(
        input_dim=(1, 28, 28),
        conv1_param={'filter_num': args.conv1_filters, 'filter_size': 3, 'pad': 1, 'stride': 1},
        conv2_param={'filter_num': args.conv2_filters, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=args.hidden_size,
        output_size=10,
        weight_init='he'
    )

    print('Model shape info:', network.model_info)

    optimizer = make_optimizer(args.optimizer, args.lr)

    train_size = x_train.shape[0]
    iter_per_epoch = int(np.ceil(train_size / args.batch_size))

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    log_lines = []

    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        idx = np.random.permutation(train_size)

        epoch_loss_sum = 0.0
        epoch_seen = 0

        for it in range(iter_per_epoch):
            batch_idx = idx[it * args.batch_size:(it + 1) * args.batch_size]
            x_batch = x_train[batch_idx]
            t_batch = t_train[batch_idx]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            epoch_loss_sum += loss * x_batch.shape[0]
            epoch_seen += x_batch.shape[0]

            if (it + 1) % 100 == 0:
                print(f'  epoch {epoch:02d}, iter {it + 1:04d}/{iter_per_epoch}, loss {loss:.4f}')

        train_acc = network.accuracy(x_train, t_train, batch_size=args.batch_size)
        test_acc = network.accuracy(x_test, t_test, batch_size=args.batch_size)
        avg_epoch_loss = epoch_loss_sum / epoch_seen

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            network.save_params('params_best.pkl')

        elapsed_epoch = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        log = (
            f'Epoch [{epoch:02d}/{args.epochs}] '
            f'Train Loss: {avg_epoch_loss:.4f} '
            f'Train Acc: {train_acc * 100:.2f}% '
            f'Test Acc: {test_acc * 100:.2f}% '
            f'Best Test Acc: {best_test_acc * 100:.2f}% '
            f'Epoch Time: {elapsed_epoch:.1f}s '
            f'Total Time: {elapsed_total / 60:.1f}min'
        )
        print(log)
        log_lines.append(log)

        # Save after every epoch so that results are not lost if training is interrupted.
        network.save_params('params_last.pkl')
        save_graphs(train_acc_list, test_acc_list, train_loss_list)
        with open('train_log.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))
        with open('last_20_log.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines[-20:]))

    print('\nTraining finished.')
    print(f'Final Train Acc: {train_acc_list[-1] * 100:.2f}%')
    print(f'Final Test Acc : {test_acc_list[-1] * 100:.2f}%')
    print(f'Best Test Acc  : {best_test_acc * 100:.2f}%')
    print('Generated files: accuracy_graph.png, loss_graph.png, train_log.txt, last_20_log.txt, params_best.pkl, params_last.pkl')


if __name__ == '__main__':
    main()
