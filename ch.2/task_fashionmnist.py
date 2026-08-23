"""
在Fashion-MNIST数据集上训练MLP做分类任务
图像简单展平，全连接处理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# 1. 加载 Fashion-MNIST
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# -------------------------
# 2. 定义 MLP 模型
# -------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # [B, 1, 28, 28] -> [B, 784]
        z = self.linear1(x)
        h = self.relu(z)
        y = self.linear2(h)
        return y


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)


# -------------------------
# 3. 损失函数和优化器
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------------------------
# 4. 训练
# -------------------------
epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {total_loss / len(train_loader):.4f}, "
        f"Train Acc: {train_acc:.4f}"
    )


# -------------------------
# 5. 测试
# -------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test Acc: {test_acc:.4f}")
