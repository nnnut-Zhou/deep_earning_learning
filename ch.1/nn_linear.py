"""
用torch创建一个随机参数的线性层，做一次推理

torch.nn.Linear(
    in_features,
    out_features,
    bias=True,
    device=None,
    dtype=None
)
参数说明
- in_features：输入样本的特征维度，也就是输入张量的最后一维
- out_features：输出样本的特征维度，也就是线性层变换后的最后一维
- bias：是否使用偏置项，默认 True；设为 False 时只学习权重，不学习偏置
- device：指定参数所在设备，例如 CPU 或 GPU
- dtype：指定参数的数据类型，例如 torch.float32
"""

import torch
import torch.nn as nn

linear_layer = nn.Linear(4, 3)

# 模拟输入数据 x，尺寸为（120，4）
x = torch.randn(120, 4)

# 调用 linear_layer
hat_y = linear_layer(x)

# 查看输出数据的尺寸
print(hat_y.shape)
