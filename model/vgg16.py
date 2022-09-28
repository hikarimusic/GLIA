import torch
from torch import nn
from torchvision import transforms

class Downsample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        return x

class Downsample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        return x

class Downsample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.maxpool(x)
        return x

class Downsample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.maxpool(x)
        return x

class Downsample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.maxpool(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer14 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer15 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer16 = nn.Sequential(
            nn.Linear(4096, 1000),
        )
    
    def forward(self, x):
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample1 = Downsample1()
        self.downsample2 = Downsample2()
        self.downsample3 = Downsample3()
        self.downsample4 = Downsample4()
        self.downsample5 = Downsample5()
        self.classifier = Classifier()
    
    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.downsample5(x)
        x = self.classifier(x)
        return x


class Utils():
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            transforms.Resize(224),
        )
        self.target_transform = None
        self.loss = nn.CrossEntropyLoss()
    
    def result(self, x):
        return torch.argmax(x, dim=1)


'''
Training Configuration in the paper
-----------------------------------

optimizer: SGD
batch size: 256
learning rate: 1e-2 -> 1e-3 -> 1e-4 -> 1e-5
momentum: 0.9
weight decay: 5e-4
data augmentation:
    RandomResizedCrop
    RandomHorizontalFlip
    ColorJitter

'''