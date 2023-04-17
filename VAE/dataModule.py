import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms

class CIFAR10_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size = 64, transform = transforms.ToTensor()) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self) -> None:
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        self.train_ds = datasets.CIFAR10(root=self.data_dir, train = True, transform=self.transform)
        self.val_ds = datasets.CIFAR10(root=self.data_dir, train = False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
