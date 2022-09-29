import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample == True:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
        if downsample == True:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x += shortcut
        x = self.relu2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample == True:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
        if downsample == True:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels//4)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels//4)
            )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels//4),
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x += shortcut
        x = self.relu3(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, True),
            BasicBlock(128, 128)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, True),
            BasicBlock(256, 256)
        )
        self.layer5 = nn.Sequential(
            BasicBlock(256, 512, True),
            BasicBlock(512, 512)
        )
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.layer5 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(64, 256),
            Bottleneck(256, 256),
            Bottleneck(256, 256)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(256, 512, True),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(512, 1024, True),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024)
        )
        self.layer5 = nn.Sequential(
            Bottleneck(1024, 2048, True),
            Bottleneck(2048, 2048),
            Bottleneck(2048, 2048)
        )
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, 1000)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(64, 256),
            Bottleneck(256, 256),
            Bottleneck(256, 256)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(256, 512, True),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(512, 1024, True),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024)
        )
        self.layer5 = nn.Sequential(
            Bottleneck(1024, 2048, True),
            Bottleneck(2048, 2048),
            Bottleneck(2048, 2048)
        )
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, 1000)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(64, 256),
            Bottleneck(256, 256),
            Bottleneck(256, 256)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(256, 512, True),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(512, 1024, True),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024)
        )
        self.layer5 = nn.Sequential(
            Bottleneck(1024, 2048, True),
            Bottleneck(2048, 2048),
            Bottleneck(2048, 2048)
        )
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, 1000)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x