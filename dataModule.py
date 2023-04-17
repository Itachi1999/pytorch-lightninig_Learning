# We will apply DataModule of PL which we will replace the Datasets and Dataloader of Pytorch


# from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F


# import torchmetrics
import pytorch_lightning as pl



# # Dataset and Dataloader
# train_ds = ImageFolder('Datasets/MNIST/train/', transform = transform)
# val_ds = ImageFolder('Datasets/MNIST/test/', transform = transform)

# train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
# val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False)



# Data Modules

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform = transforms.ToTensor(), batch_size = 64) -> None:
        super().__init__()
        self.data_dir = data_dir
        # self.num_workers = num_workers
        self.bs = batch_size
        self.transform = transform

    def prepare_data(self) -> None:
        # Download Dataset
        datasets.KMNIST(self.data_dir, train = True, download = True)
        datasets.KMNIST(self.data_dir, train = False, download = True)

    def setup(self, stage: str) -> None:
        # Random split and initialising them
        # If you have dataset already downloaded then you skip prepare_data and use CustomDataset here in this function

        entire_ds = datasets.KMNIST(self.data_dir, train=True, download=False, transform = self.transform)
        self.train_ds, self.val_ds = random_split(entire_ds, [50000, 10000])
        self.test_ds = datasets.KMNIST(self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, shuffle=False)


