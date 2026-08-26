"""
以torch实现经典CNN：vgg
相对于大小核交替的alexnet，它固定卷积核为3x3，采取更深的网络层数
"""

import torch
import torch.nn as nn
from torchsummary import summary

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        """
        这里的batchnorm2d层用于对图像做归一化
        nn.BatchNorm2d(num_features)
        num_features是输入图的通道数
        它会跨批次对每个通道做归一化，具体计算需要理解：
        比如[n, c, h, w]，则对于每个c里每个像素，都需要计算n张照片中同处一个c里的所有像素，做归一化
        """
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg["A"], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg["B"], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg["D"], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg["E"], batch_norm=True))


if __name__ == "__main__":
    net = vgg19_bn()
    print(summary(net, (3, 224, 224), batch_size=2))
