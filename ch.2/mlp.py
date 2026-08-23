"""
创建一个简单的多层感知机，做一次前向传播
在线性层基础上添加ReLU激活层

nn.ReLU不会改变张量的形状，针对每一个元素分别计算，输入和输出的形状一致
"""

import torch
import torch.nn as nn

# 1. 输入：一个样本，包含三个特征
x = torch.randn(1, 3)  # shape: [batch_size, 3]


# 2. 模型：3 输入 -> 3 隐藏神经元 -> ReLU -> 2 输出
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        z = self.fc1(x)  # shape: [batch_size, 3]
        h = self.relu(z)  # shape: [batch_size, 3]
        out = self.fc2(h)  # shape: [batch_size, 2]
        return out


model = MLP()


# 3. 输出：前向传播
output = model(x)

print(output)
print(output.shape)
