"""
以torch实现经典cnn：alexnet
"""

import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        """
        nn.Sequential(...)把每个layer串起来顺序执行

        nn.Sequential(layer1,layer2,layer3)
        相当于
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),  # 注意dropout层，随机将50%神经元输出置0。只在train()时生效
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = AlexNet()
    print(summary(net, (3, 224, 224), batch_size=2))
