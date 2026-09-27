"""
使用torchinfo.summary()函数查看网络结构和参数量
"""

from torchvision import models
import torchinfo

net = models.alexnet(weights=None)
torchinfo.summary(net, (1, 3, 224, 224))
