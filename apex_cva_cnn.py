import torch
import torch.nn as nn
from torchvision import models

class ApexCVANet(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = b.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(576, 128), nn.Hardswish(), nn.Linear(128, 10), nn.Sigmoid())
    def forward(self, x):
        return self.head(self.pool(self.features(x)))
