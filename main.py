import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import models.ResNet as rn
from task import *


def load_FMNIST():
    _training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    _test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    _labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    return _training_data, _test_data, _labels_map


if __name__ == '__main__':
    training_data, test_data, labels_map = load_FMNIST()
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # training from scratch
    model = rn.ResNet18().to(device)

    # loading from checkpoint
    model.load_state_dict(torch.load("./checkpoints/test.pth"))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    learning_rate = 1e-3

    #trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)

    task = Task(model, train_dataloader, test_dataloader, labels_map, batch_size
                , loss_fn, epochs, device, learning_rate)
    task.train()
    # task.visualize_incorrect()
