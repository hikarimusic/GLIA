import torch
from torchvision.io import read_image

from model import MLP

INPUT_IMAGE = ["./data/test/image/0.png", "./data/test/image/1.png", "./data/test/image/2.png"]
CLASS_NAME = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LOAD_WEIGHT = "weight.pth"

def predict(image, model):
    for image in INPUT_IMAGE:
        image = read_image(image)
        image = image.float()
        image = image[None,:]
        pred = model(image)
        result = CLASS_NAME[torch.argmax(pred, 1).item()]
        print(result)
    

if __name__ == '__main__' :
    model = MLP()
    if LOAD_WEIGHT:
        model.load_state_dict(torch.load(LOAD_WEIGHT))
    predict(INPUT_IMAGE, model)