# PyTorch Fashion MNIST CNN

This folder contains the PyTorch implementation for Fashion MNIST CNN classification.

## 파일

- train_pytorch_fashion_mnist.py: PyTorch CNN training source code
- accuracy_graph.png: train/test accuracy graph
- loss_graph.png: train/test loss graph
- train_log.txt: full training log
- last_20_log.txt: last 20 lines of training log
- fashion_cnn_best.pth: best model parameters
- fashion_cnn_last.pth: last epoch model parameters

## Dataset

Fashion MNIST dataset 은 깃허브 파일 용량 제한 25MB를 넘겨서 미처 첨부하지 못 했습니다.

- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

## 학습 결과

Final Train Accuracy: 96.30%
Final Test Accuracy: 93.96%
Best Test Accuracy: 93.96%

