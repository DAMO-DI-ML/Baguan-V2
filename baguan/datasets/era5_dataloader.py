from baguan.utils import Args
from baguan.datasets import ERA5Datasets

from torch.utils.data import DataLoader
from lightning import LightningDataModule


class ERA5DataLoader(LightningDataModule):
    def __init__(self, args=Args()):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.params = args

    def setup(self, stage=None):
        self.train_dataset = ERA5Datasets(root=self.params.root, flag='train')
        self.val_dataset = ERA5Datasets(root=self.params.root, flag='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.params.data.__dict__, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.params.data.__dict__, shuffle=False)

