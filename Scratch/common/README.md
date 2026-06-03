Scratch Fashion MNIST CNN 실행 안내
=================================

1. 이 폴더(Scratch)에 아래 Fashion MNIST 원본 파일 4개를 복사합니다.
   - train-images-idx3-ubyte.gz
   - train-labels-idx1-ubyte.gz
   - t10k-images-idx3-ubyte.gz
   - t10k-labels-idx1-ubyte.gz

2. Scratch 폴더 구조는 아래처럼 둡니다.

Scratch/
├─ train_scratch_fashion_mnist.py
├─ fashion_mnist_loader.py
├─ fashion_convnet.py
├─ common/
│  ├─ functions.py
│  ├─ gradient.py
│  ├─ layers.py
│  ├─ optimizer.py
│  └─ util.py
├─ train-images-idx3-ubyte.gz
├─ train-labels-idx1-ubyte.gz
├─ t10k-images-idx3-ubyte.gz
└─ t10k-labels-idx1-ubyte.gz

3. 빠른 동작 확인용 실행입니다. 정확도 제출용이 아니라 오류 확인용입니다.

py train_scratch_fashion_mnist.py --epochs 1 --max_train 1000 --max_test 1000

4. 제출용 전체 학습 실행입니다.

py train_scratch_fashion_mnist.py --epochs 35

5. 정확도가 부족하면 아래처럼 epoch를 늘립니다.

py train_scratch_fashion_mnist.py --epochs 50

6. 생성 파일
   - accuracy_graph.png: 훈련/테스트 정확도 그래프
   - loss_graph.png: 훈련 loss 그래프
   - train_log.txt: 전체 로그
   - last_20_log.txt: 마지막 20줄 로그
   - params_best.pkl: 최고 테스트 정확도 파라미터
   - params_last.pkl: 마지막 epoch 파라미터

주의:
- Scratch 파트에는 torch, torchvision, tensorflow, keras를 import하지 않습니다.
- common/layers.py의 Convolution, Pooling, Affine, Relu, SoftmaxWithLoss를 사용합니다.
- common/util.py의 im2col, col2im을 사용합니다.
