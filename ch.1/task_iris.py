"""
使用IRIS数据进行分类任务的模型训练
https://www.kaggle.com/datasets/arshid/iris-flower-dataset
固定训练500轮
"""

import pandas as pd
import random
import torch
import torch.nn as nn


def load_iris_csv(csv_path):
    data = pd.read_csv(csv_path)

    if data.shape[1] < 5:
        raise ValueError("CSV should contain 4 feature columns and 1 label column.")

    feature_columns = data.columns[:4]
    label_column = data.columns[-1]

    features = data[feature_columns].values
    labels = data[label_column].values

    label_names = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    targets = [label_to_id[label] for label in labels]

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)

    return x, y, label_names


def train_test_split(x, y, test_ratio=0.2, seed=42):
    random.seed(seed)

    indices = list(range(x.size(0)))
    random.shuffle(indices)

    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def main():
    torch.manual_seed(42)

    # 数据准备
    CSV_PATH = r"./data/IRIS.csv"
    x, y, label_names = load_iris_csv(CSV_PATH)
    x_train, y_train, x_test, y_test = train_test_split(x, y)

    # 对输入数据做 Normalization
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    # 超参数设置
    num_features = x_train.shape[-1]
    num_classes = len(label_names)
    lr = 0.1
    epochs = 500

    # 定义模型
    model = nn.Linear(num_features, num_classes)

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(epochs):
        logits = model(x_train)
        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            train_acc = accuracy(logits, y_train)
            print(
                f"epoch {epoch + 1:3d} | loss {loss.item():.4f} | train_acc {train_acc:.4f}"
            )

    # 测试
    test_logits = model(x_test)
    test_acc = accuracy(test_logits, y_test)

    print("\nTest results")
    print(f"test_acc:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
