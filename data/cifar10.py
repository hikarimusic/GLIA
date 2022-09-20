import os
import pickle
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil
from torch.utils.data import Dataset


def download():
    if not os.path.exists(os.path.join(os.getcwd(), "cifar-10-python.tar.gz")):
        os.system("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        os.system("tar -xvzf cifar-10-python.tar.gz")
    if not os.path.exists(os.path.join(os.getcwd(), "cifar10")):
        os.mkdir(os.path.join(os.getcwd(), "cifar10"))
    if not os.path.exists(os.path.join(os.getcwd(), "cifar10", "train")):
        os.mkdir(os.path.join(os.getcwd(), "cifar10", "train"))
    if not os.path.exists(os.path.join(os.getcwd(), "cifar10", "test")):
        os.mkdir(os.path.join(os.getcwd(), "cifar10", "test"))

    train = pd.DataFrame(columns=["image", "label"])
    batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    for batch in batches:
        with open(os.path.join(os.getcwd(), "cifar-10-batches-py", batch), 'rb') as fo:
            print(f"Converting {batch}")
            cifar = pickle.load(fo, encoding='bytes')
            for id in tqdm(range(len(cifar[b'data']))):
                arr = cifar[b'data'][id].reshape((3, 32, 32)).transpose(1, 2, 0)
                img = Image.fromarray(arr, "RGB")
                img.save(os.path.join(os.getcwd(), "cifar10", "train", cifar[b'filenames'][id].decode()))
            table = pd.DataFrame({
                "image" : [x.decode() for x in cifar[b'filenames']],
                "label" : cifar[b'labels']
            })
            train = pd.concat([train, table], ignore_index=True)
    train.to_csv(os.path.join(os.getcwd(), "cifar10", "train.csv"), index=False)

    test = pd.DataFrame(columns=["image", "label"])
    with open(os.path.join(os.getcwd(), "cifar-10-batches-py", "test_batch"), 'rb') as fo:
        print(f"Converting {batch}")
        cifar = pickle.load(fo, encoding='bytes')
        for id in tqdm(range(len(cifar[b'data']))):
            arr = cifar[b'data'][id].reshape((3, 32, 32)).transpose(1, 2, 0)
            img = Image.fromarray(arr, "RGB")
            img.save(os.path.join(os.getcwd(), "cifar10", "test", cifar[b'filenames'][id].decode()))
        table = pd.DataFrame({
            "image" : [x.decode() for x in cifar[b'filenames']],
            "label" : cifar[b'labels']
        })
        test = pd.concat([test, table], ignore_index=True)
    test.to_csv(os.path.join(os.getcwd(), "cifar10", "test.csv"), index=False)

    with open(os.path.join(os.getcwd(), "cifar-10-batches-py", "batches.meta"), 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        classes = [x.decode() for x in data[b'label_names']]
        with open(os.path.join(os.getcwd(), "cifar10", "class.txt"), 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
    
    os.remove(os.path.join(os.getcwd(), "cifar-10-python.tar.gz"))
    shutil.rmtree(os.path.join(os.getcwd(), "cifar-10-batches-py"))


class TrainDataset(Dataset):
    def __init__(self):
        self.table = pd.read_csv(os.path.join(os.getcwd(), "cifar10", "train.csv"))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_path = os.path.join(os.getcwd(), "cifar10", "train", self.table["image"][idx])
        img = Image.open(img_path)
        lbl = self.table["label"][idx]
        return img, lbl


if __name__ == '__main__':
    download()



