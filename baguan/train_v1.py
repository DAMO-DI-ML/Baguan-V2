import numpy as np
import torch
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torchvision.transforms import transforms

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from baguan.utils import ArgsV1
from baguan.utils import get_dataloader
from baguan.modules import BaguanV1Module
from baguan.datasets.era5_dataloader import ERA5DataLoader


def main(args=ArgsV1()):
    seed_everything(args.seed, workers=True)

    # strategy = 'ddp'
    strategy = FSDPStrategy()
    # strategy = DeepSpeedStrategy(stage=2)
    
    proj_name = 'baguan_v1_finetune_24h_ct'

    callbacks = [
        ModelCheckpoint(
            dirpath=f'/jupyter/BaguanV1/checkpoint/{proj_name}',
            save_top_k=100,
            save_last=True,
            monitor='train/loss',
            filename="epoch_{epoch:03d}"
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    logger = WandbLogger(
        name=proj_name,
        save_dir=f"./logs/baguan_v1/{proj_name}",
        project='baguanv1',
        entity='maziqing_team',
    )

    trainer = Trainer(
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=True,
        **args.trainer.__dict__
    )

    train_dataloader = get_dataloader(flag='train', args=args)
    val_dataloader = get_dataloader(flag='val', args=args)
    
    len_const_vars = len(train_dataloader.dataset.constant_vars)
    len_inp_vars = len(train_dataloader.dataset.in_surface_ids) + len(train_dataloader.dataset.in_upper_ids)
    len_out_vars = len(train_dataloader.dataset.out_surface_ids) + len(train_dataloader.dataset.out_upper_ids)
    out_surface_vars = train_dataloader.dataset.out_surface_vars
    out_upper_vars_all = train_dataloader.dataset.out_upper_vars_all
    const_vars = train_dataloader.dataset.constant_vars

    # denorm transform
    surface_mean_norm = train_dataloader.dataset.out_surface_transforms.mean
    surface_std_norm = train_dataloader.dataset.out_surface_transforms.std
    surface_mean_denorm, surface_std_denorm = -surface_mean_norm / surface_std_norm, 1 / surface_std_norm
    upper_mean_norm = train_dataloader.dataset.out_upper_transforms.mean
    upper_std_norm = train_dataloader.dataset.out_upper_transforms.std
    upper_mean_denorm, upper_std_denorm = -upper_mean_norm / upper_std_norm, 1 / upper_std_norm
    transform_denorm = transforms.Normalize(
        np.hstack([surface_mean_denorm, upper_mean_denorm]), 
        np.hstack([surface_std_denorm, upper_std_denorm])
    )
    
    model = BaguanV1Module(
        args, 
        len_const_vars, len_inp_vars, len_out_vars, 
        out_surface_vars, out_upper_vars_all, const_vars,
        transform_denorm, 
        pretrained_path=args.pretrained_path
    )
    datamodule = ERA5DataLoader(args)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()