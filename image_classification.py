from data.cifar10 import Data
from model.vgg16 import Model, Utils

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchsummary import summary


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_WEIGHT = os.path.join(os.getcwd(), "weight", "vgg.pth")
SAVE_WEIGHT = os.path.join(os.getcwd(), "weight", "vgg.pth")

INPUT_CHANNEL = 3
INPUT_SIZE = 224
DATA_AUGMENT = torch.nn.Sequential(
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
)

BATCH_SIZE = 16
EPOCHS = 10
OPTIMIZER = torch.optim.SGD
LEARN_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


INPUT_IMAGE = os.path.join(os.getcwd(), "data", "cifar10", "test", "abandoned_ship_s_000635.png")


def train(data, model, utils):
    print("\nTraining")
    size = len(data.dataset)
    avg_loss = 0
    for batch, (X, y) in enumerate(data):
        X, y = X.to(DEVICE), y.to(DEVICE)
        X = utils.transform(X) if utils.transform else X
        X = DATA_AUGMENT(X)
        _y = model(X)
        y = utils.target_transform(y) if utils.target_transform else y
        loss = utils.loss(_y, y)
        optimizer = OPTIMIZER(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        print(f"\rloss: {avg_loss/(batch%100+1):>7f}  [{(batch+1) * len(X):>5d}/{size:>5d}]", end="")
        if (batch+1) % 100 == 0:
            avg_loss = 0
            print("")
    print("")


def test(data, model, utils):
    print("\nTesting")
    size = len(data.dataset)
    num_batches = len(data)
    avg_loss, avg_corr = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(data):
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = utils.transform(X) if utils.transform else X
            _y = model(X)
            y = utils.target_transform(y) if utils.target_transform else y
            pred = utils.result(_y)
            avg_loss += utils.loss(_y, y).item()
            avg_corr += (pred == y).type(torch.float).sum().item()
            print(f"\rloss: {avg_loss/(batch+1):>7f}  [{(batch+1) * len(X):>5d}/{size:>5d}]", end="")
    avg_corr /= size
    print(f"\nAccuracy: {(100*avg_corr):>0.1f}%")


def use(image, data, model, utils):
    print("\nUse the model")
    img = Image.open(image)
    img = data.transform(img) if data.transform else X
    X = img.unsqueeze(0)
    X = X.to(DEVICE)
    X = utils.transform(X) if utils.transform else X
    _y = model(X)
    pred = utils.result(_y)
    print(f"Image: {str(image)}")
    print(f"Predict: {data.classes[pred]}")


if __name__ == '__main__':
    data = Data()
    train_data = DataLoader(data.train, batch_size=BATCH_SIZE)
    test_data = DataLoader(data.test, batch_size=BATCH_SIZE)
    model = Model()
    model = model.to(DEVICE)
    utils = Utils()
    if LOAD_WEIGHT:
        model.load_state_dict(torch.load(LOAD_WEIGHT))
    summary(model, (INPUT_CHANNEL, INPUT_SIZE, INPUT_SIZE))

    for t in range(EPOCHS):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_data, model, utils)
        test(test_data, model, utils)
        if SAVE_WEIGHT:
            torch.save(model.state_dict(), SAVE_WEIGHT)
                      
    use(INPUT_IMAGE, data, model, utils)    


