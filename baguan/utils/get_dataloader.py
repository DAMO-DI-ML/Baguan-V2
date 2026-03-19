from torch.utils.data import DataLoader

from baguan.utils import Args
from baguan.datasets import ERA5Datasets


def get_dataloader(flag='train', args=Args()):
    shuffle = True if flag == 'train' else False
    datasets = ERA5Datasets(root=args.root, flag=flag)
    dataloader = DataLoader(datasets, **args.data.__dict__)
    return dataloader