"""
使用Netron可视化ONNX模型
"""

import torch
import torch.nn as nn
import netron
import onnx
from onnx import shape_inference


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


net = My_Net()
img = torch.rand((1, 3, 224, 224))
torch.onnx.export(
    model=net,
    args=img,
    f="model.onnx",
    input_names=["image"],
    output_names=["feature_map"],
    training=torch.onnx.TrainingMode.TRAINING,
)
onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")

netron.start("model.onnx")

input = input("Press any key to exit...")
