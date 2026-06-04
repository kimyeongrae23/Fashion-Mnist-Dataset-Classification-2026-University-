Scratch Fashion MNIST CNN

이 폴더는 Fashion MNIST 이미지 분류를 위한 Scratch CNN 구현 코드입니다.

본 구현에서는 PyTorch, TensorFlow, Keras와 같은 기존 딥러닝 프레임워크 API를 사용하지 않았습니다.
교재 『밑바닥부터 시작하는 딥러닝』의 CNN 관련 소스코드를 응용하여 NumPy 기반으로 구현했습니다.

1. 구현 개요

본 모델은 교재에서 제공하는 CNN 계층 구조를 기반으로 구현했습니다.

사용한 주요 구성 요소는 다음과 같습니다.

im2col
col2im
Convolution
Pooling
Relu
Affine
SoftmaxWithLoss
Adam optimizer

기본 SimpleConvNet 구조를 바탕으로 하되, Fashion MNIST 분류 성능을 높이기 위해 합성곱 계층을 추가했습니다.

2. 모델 구조

입력 데이터 형상은 다음과 같습니다.

(N, 1, 28, 28)

CNN 구조는 다음과 같습니다.

Input
→ Convolution
→ ReLU
→ Pooling
→ Convolution
→ ReLU
→ Pooling
→ Affine
→ ReLU
→ Affine
→ SoftmaxWithLoss

Fashion MNIST는 숫자 MNIST보다 분류 난이도가 높고, 셔츠, 코트, 풀오버처럼 서로 형태가 비슷한 클래스가 존재합니다.
따라서 특징 추출 능력을 높이기 위해 기본 CNN 구조보다 합성곱 계층을 추가하여 구현했습니다.

3. 파일 구성
Scratch/
train_scratch_fashion_mnist.py
fashion_mnist_loader.py
fashion_convnet.py
verify_no_framework.py
common/
__init__.py
functions.py
gradient.py
layers.py
optimizer.py
util.py
accuracy_graph.png
loss_graph.png
train_log.txt
last_20_log.txt
params_best.pkl
params_last.pkl

5. 데이터셋

Fashion MNIST 데이터 파일은 용량 문제로 저장소에 포함하지 않았습니다.

아래 4개의 Fashion MNIST 데이터 파일을 공식 링크에서 다운로드한 뒤, Scratch 폴더 안에 넣고 실행하면 됩니다.

train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
5. 실행 방법

먼저 PyTorch, TensorFlow, Keras와 같은 딥러닝 프레임워크가 import되지 않았는지 확인합니다.

py verify_no_framework.py

정상 출력 예시는 다음과 같습니다.

OK: no PyTorch/TensorFlow/Keras import statement found.

그 다음 Scratch CNN 학습 코드를 실행합니다.

py train_scratch_fashion_mnist.py --epochs 35

코드 오류 확인용으로 빠르게 실행하고 싶을 경우에는 일부 데이터만 사용하여 실행할 수 있습니다.

py train_scratch_fashion_mnist.py --epochs 1 --max_train 1000 --max_test 1000
6. 학습 결과

총 35 epoch 동안 학습을 진행했습니다.

최종 결과는 다음과 같습니다.

Final Train Accuracy: 99.81%
Final Test Accuracy : 92.09%
Best Test Accuracy  : 92.42%

과제에서 요구한 기준은 다음과 같습니다.

훈련 정확도 95% 이상
테스트 정확도 92% 이상

본 Scratch CNN 모델은 최종 훈련 정확도 99.81%, 최종 테스트 정확도 92.09%, 최고 테스트 정확도 92.42%를 기록하여 과제 기준을 만족했습니다.

7. 출력 결과 파일

학습이 끝나면 다음 결과 파일들이 생성됩니다.

accuracy_graph.png
loss_graph.png
train_log.txt
last_20_log.txt
params_best.pkl
params_last.pkl

accuracy_graph.png는 훈련 정확도와 테스트 정확도의 변화를 나타냅니다.
loss_graph.png는 훈련 손실과 테스트 손실의 변화를 나타냅니다.
last_20_log.txt는 학습 로그의 마지막 20줄을 저장한 파일입니다.
