# train_pytorch_fashion_mnist.py

import os
import gzip
import struct
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FashionMNISTFromGZ(Dataset):
    def __init__(self, image_path, label_path, train=True, augment=False):
        self.images = self.load_images(image_path)
        self.labels = self.load_labels(label_path)
        self.train = train
        self.augment = augment

        if len(self.images) != len(self.labels):
            raise ValueError("이미지 개수와 라벨 개수가 다릅니다.")

    def load_images(self, path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"이미지 파일 magic number 오류: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num, 1, rows, cols)
            data = data.astype(np.float32) / 255.0
        return data

    def load_labels(self, path):
        with gzip.open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"라벨 파일 magic number 오류: {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def random_horizontal_flip(self, image):
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
        return image

    def random_crop_with_padding(self, image, padding=2):
        padded = np.pad(
            image,
            ((0, 0), (padding, padding), (padding, padding)),
            mode="constant"
        )

        _, h, w = padded.shape
        top = random.randint(0, h - 28)
        left = random.randint(0, w - 28)

        cropped = padded[:, top:top + 28, left:left + 28]
        return cropped.copy()

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.train and self.augment:
            image = self.random_crop_with_padding(image, padding=2)
            image = self.random_horizontal_flip(image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        image = (image - 0.5) / 0.5

        return image, label


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    required_files = [
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"파일이 없습니다: {file}")

    train_dataset = FashionMNISTFromGZ(
        args.train_images,
        args.train_labels,
        train=True,
        augment=True
    )
    
    train_eval_dataset = FashionMNISTFromGZ(
        args.train_images,
        args.train_labels,
        train=False,
        augment=False
    )


    test_dataset = FashionMNISTFromGZ(
        args.test_images,
        args.test_labels,
        train=False,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = FashionCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    log_lines = []

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_aug_loss = running_loss / total
        
        train_aug_acc = correct / total

        train_eval_loss, train_eval_acc = evaluate(
            model,
            train_eval_loader,
            criterion,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        train_loss_list.append(train_eval_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_eval_acc)
        test_acc_list.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "fashion_cnn_best.pth")

        log = (
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_eval_loss:.4f} "
            f"Train Acc: {train_eval_acc * 100:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc * 100:.2f}% "
            f"Aug Train Acc: {train_aug_acc * 100:.2f}% "
            f"Best Test Acc: {best_test_acc * 100:.2f}%"
        )   

        print(log)
        log_lines.append(log)

    torch.save(model.state_dict(), "fashion_cnn_last.pth")

    with open("train_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    with open("last_20_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines[-20:]))

    epochs = range(1, args.epochs + 1)

    plt.figure()
    plt.plot(epochs, train_acc_list, label="train acc")
    plt.plot(epochs, test_acc_list, label="test acc", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_graph.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss_list, label="train loss")
    plt.plot(epochs, test_loss_list, label="test loss", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_graph.png", dpi=200)
    plt.show()

    print()
    print("학습 완료")
    print(f"최고 테스트 정확도: {best_test_acc * 100:.2f}%")
    print("생성 파일: accuracy_graph.png, loss_graph.png, train_log.txt, last_20_log.txt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_images", type=str, default="train-images-idx3-ubyte.gz")
    parser.add_argument("--train_labels", type=str, default="train-labels-idx1-ubyte.gz")
    parser.add_argument("--test_images", type=str, default="t10k-images-idx3-ubyte.gz")
    parser.add_argument("--test_labels", type=str, default="t10k-labels-idx1-ubyte.gz")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()