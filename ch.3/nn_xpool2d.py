"""
演示三种池化层的用法
"""

import torch
import torch.nn as nn

x = torch.tensor(
    [
        [
            [
                [1.0, 2.0, 5.0, 4.0],
                [3.0, 8.0, 6.0, 7.0],
                [2.0, 1.0, 0.0, 3.0],
                [4.0, 6.0, 2.0, 9.0],
            ]
        ]
    ]
)


"""
最大池化层
pool = nn.MaxPool2d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False
)
参数 ：
- kernel_size：池化窗口大小x
  - kernel_size=2 等价于 kernel_size=(2, 2)
  - kernel_size=(2, 3) 表示窗口高度为 2，宽度为 3
- stride：池化窗口滑动步幅
  - 默认值为 None，当 stride=None 时，默认 stride = kernel_size
- padding：输入特征图边缘补边大小
  - padding=0：不补边
  - padding=1：上下左右各补 1 圈
  - 对 MaxPool2d 来说，补边区域会被看作负无穷，因此一般不会成为最大值
- ceil_mode：输出尺寸是否使用向上取整。
  - 默认值为 False，使用向下取整
  - 设置为 True 时，输出尺寸可能更大
"""

pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

y1 = pool1(x)

print("input shape:", x.shape)
print("output shape:", y1.shape)
print(y1)

"""
平均池化层

pool = nn.AvgPool2d(
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
)
参数 ：
- kernel_size：池化窗口大小。
  - kernel_size=2 等价于 kernel_size=(2, 2)
  - kernel_size=(2, 3) 表示窗口高度为 2，宽度为 3
- stride：池化窗口滑动步幅。
  - 默认值为 None
  - 当 stride=None 时，默认 stride = kernel_size
- padding：输入特征图边缘补边大小。
  - padding=0：不补边
  - padding=1：上下左右各补 1 圈
  - 对 AvgPool2d 来说，补边值通常为 0
- ceil_mode：输出尺寸是否使用向上取整。
  - 默认值为 False
  - 设置为 True 时，输出尺寸可能更大
"""

pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

y2 = pool2(x)

print("input shape:", x.shape)
print("output shape:", y2.shape)
print(y2)


"""
全局平均池化层

pool = nn.AdaptiveAvgPool2d(
    output_size
)
- output_size：指定输出特征图的空间尺寸。
  - output_size=1：等价于 output_size=(1, 1)
  - output_size=(4, 4)：输出空间尺寸为 4 × 4
  - output_size=(None, 7)：高度保持输入大小，宽度变为 7
"""

pool3 = nn.AdaptiveAvgPool2d((1, 1))

y3 = pool3(x)

print("input shape:", x.shape)
print("output shape:", y3.shape)
