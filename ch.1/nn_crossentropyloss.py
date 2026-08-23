"""
用模拟的logits和结果计算交叉熵损失

torch.nn.CrossEntropyLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean',
)
参数说明
- weight：类别权重，形状为 C（C 是类别数量）；常用于类别不均衡的情况
- ignore_index：指定某个标签值不参与损失计算，常用于语义分割或序列任务中的 padding
- reduction：对于一个 batch 中的多个样本，损失值的聚合方式，默认 'mean'
  - 'none'：不做聚合，返回每个样本的 loss
  - 'mean'：返回所有样本 loss 的平均值
  - 'sum'：返回所有样本 loss 的总和
"""

import torch
import torch.nn as nn

# 模拟模型输出 logits，尺寸为（120，3）
# 120 表示样本数量，3 表示类别数量
logits = torch.randn(120, 3)

# 模拟真实标签，尺寸为（120）
# 每个标签的取值范围是 [0, 3)，也就是 0、1、2
labels = torch.randint(0, 3, (120,), dtype=torch.long)

# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 计算损失
loss = loss_fn(logits, labels)

# 查看损失值
print(loss.shape)
