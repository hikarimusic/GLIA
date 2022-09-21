from data.cifar10 import TrainDataset, TestDataset
from model.vgg16 import Model, Utils

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_WEIGHT = None
SAVE_WEIGHT = os.path.join(os.getcwd(), "weight", "vgg.pth")

BATCH_SIZE = 64
EPOCHS = 20
OPTIMIZER = torch.optim.SGD
LEARN_RATE = 1e-3

def train(data, model, utils):
    print("Training")
    model.train()
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        X, y = X.to(DEVICE), y.to(DEVICE)
        X = utils.transform(X) if utils.transform else X
        y = utils.target_transform(y) if utils.target_transform else y
        _y = model(X)
        loss = utils.loss(_y, y)
        optimizer = OPTIMIZER(model.parameters(), lr=LEARN_RATE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(data, model, utils):
    print("Testing")
    model.eval()
    size = len(data.dataset)
    num_batches = len(data)
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(data):
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = utils.transform(X) if utils.transform else X
            y = utils.target_transform(y) if utils.target_transform else y
            _y = model(X)
            pred = utils.result(_y)
            loss += utils.loss(_y, y).item()
            correct += (pred == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg. Loss: {loss:>8f} \n")


if __name__ == '__main__':
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = Model()
    utils = Utils()
    if LOAD_WEIGHT:
        model.load_state_dict(torch.load(LOAD_WEIGHT))

    for t in range(EPOCHS):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_data, model, utils)
        test(test_data, model, utils)
        if SAVE_WEIGHT:
            torch.save(model.state_dict(), SAVE_WEIGHT)
