from torch import nn
import torch.nn.functional as F

from .resnet_custom import resnet18, resnet32_grasp
from .wide_resnet import Wide_ResNet
from torchvision.models import vgg


class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        return x


class DeepModel(nn.Module):
    def __init__(self, input_size, num_classes, config):
        super().__init__()

        # assert input_size in [32, 64], "Imagenet is not supported yet"

        if config == "resnet18":
            self.model = resnet18(num_classes=num_classes)
        elif config == "resnet32grasp":
            self.model = resnet32_grasp(num_classes=num_classes)
        elif config == "wideresnet":
            dropout_rate = 0.0
            # self.model = Wide_ResNet(28, 10, dropout_rate, num_classes)
            self.model = Wide_ResNet(40, 10, dropout_rate, num_classes)
        elif config == "vgg19":
            # Adapted from:
            # https://github.com/alecwangcq/GraSP/blob/master/models/base/vgg.py
            cfg = [
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
            ]
            self.model = vgg.VGG(
                vgg.make_layers(cfg, batch_norm=True), num_classes=num_classes
            )
            self.model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.model.classifier = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        else:
            raise TypeError

    def forward(self, x):
        x = self.model(x)

        return x
