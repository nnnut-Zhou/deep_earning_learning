"""
卷积运算演示，3通道图像转64通道输出

conv = nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    padding_mode='zeros'
)
参数 ：
  - in_channels ：输入特征图的通道数
  - out_channels ：输出特征图的通道数
  - kernel_size ：卷积核大小。
    - 如果写成一个整数，比如 kernel_size=3，等价于 kernel_size=(3, 3)
    - 也可以写成不同的高宽 ：kernel_size=(3, 5)，表示卷积核高度为 3，宽度为 5
  - stride ：卷积核滑动步幅
  - padding ：输入特征图边缘补多少圈像素，即 单侧补充大小，默认用 0 补边
    - padding=0，默认值，表示 不补边
    - padding=1 ：四周补 1 圈，上下左右都补 1 圈
    - padding=(1, 2) ：  (上下补多少, 左右补多少) ，即 ：top = 1、bottom = 1、left = 2、right = 2
    - 如果你想分别控制左、右、上、下，需要先用 torch.nn.functional.pad 手动补边
  - bias ：是否添加偏置项，值为 True 或 False
  - padding_mode ：补边方式 ：
    - padding_mode='zeros' ： 表示用 0 补边
    - 'reflect' ：镜像补边，不重复边界值
    - 'replicate' ：镜像补边，不重复边界值
    - 'circular' ：循环补边
"""

import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 8 张 32×32 像素的 RGB 随机噪声图像
x = torch.randn(8, 3, 32, 32)
y = conv(x)

print(y.shape)
