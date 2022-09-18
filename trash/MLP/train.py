import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision.io import read_image
import pandas as pd

from model import MLP

TRAIN_IMAGE = "./data/train/image"
TRAIN_LABEL = "./data/train/label.csv"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD
LEARN_RATE = 1e-3
LOAD_WEIGHT = None
SAVE_WEIGHT = "weight.pth"

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def data_load():
    data = CustomDataset(annotations_file=TRAIN_LABEL, img_dir=TRAIN_IMAGE)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def train(data, model):
    size = len(data.dataset)
    model.train()
    for batch, (X, y) in enumerate(data):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = LOSS_FN(pred, y)
        optimizer = OPTIMIZER(model.parameters(), lr=LEARN_RATE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def eval(data, model):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss += LOSS_FN(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Eval Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

if __name__ == '__main__' :
    data = data_load()
    model = MLP()
    if LOAD_WEIGHT:
        model.load_state_dict(torch.load(LOAD_WEIGHT))
    print(f"Using {DEVICE} device")
    for t in range(EPOCHS):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(data, model)
        eval(data, model)
    if SAVE_WEIGHT:
        torch.save(model.state_dict(), SAVE_WEIGHT)

