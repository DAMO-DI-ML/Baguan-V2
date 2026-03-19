# Baguan-V2

A deep learning framework for global weather forecasting using Swin Transformer V2 architecture.

## Overview

Baguan-V2 is a PyTorch-based weather forecasting model that leverages the Swin Transformer V2 architecture with cross-resolution attention mechanisms. It is designed for medium-range global weather prediction using ERA5 reanalysis data.

## Architecture

### Core Model (`baguan/models/baguan_v2.py`)

The main model architecture consists of:

- **HieraWeatherEmbedding**: Hierarchical weather variable embedding layer
- **SwinTransformerV2CrStage**: Swin Transformer V2 with cross-resolution stages
- **Multi-head Attention**: 16 attention heads with 1536 embedding dimensions
- **Patch Size**: 6x6 spatial patches on 720x1440 latitude-longitude grids
- **Depth**: 24 transformer layers
- **Window Size**: Dynamic window sizing based on image-window ratio (72)

Key features:
- Full positional embeddings for spatial information
- Date and hour embeddings for temporal encoding
- Residual connections for stable training
- Gradient checkpointing for memory efficiency

### Alternative Models

- **BaguanV1** (`baguan/models/baguan_v1.py`): MAE-based ClimaX architecture with variable tokenization
- **Fuxi** (`baguan/models/fuxi.py`): 3D cube embedding with Swin Transformer stages

## Training

### Main Training Script (`baguan/train.py`)

The training pipeline uses PyTorch Lightning with:

- **Strategy**: FSDP (Fully Sharded Data Parallel)
- **Logger**: Weights & Biases (wandb)
- **Checkpointing**: Top-k model saving with loss monitoring
- **Mixed Precision**: 16-bit mixed precision training

```bash
python -m baguan.train
```

### Training Module (`baguan/modules/modules.py`)

`BaguanV2Module` implements:
- Autoregressive training with multi-step rollout
- Latitude-weighted MAE and RMSE metrics
- AdamW optimizer with weight decay separation
- Linear warmup cosine annealing learning rate scheduler

### Data Pipeline (`baguan/datasets/`)

**ERA5Datasets** (`era5_dataset.py`):
- Supports surface and upper-air variables
- 13 pressure levels (50-1000 hPa)
- 5 upper variables: z, q, t, u, v
- 16 surface variables including precipitation (tp1h, tp6h)
- 6 constant variables (orography, land-sea mask, etc.)
- Normalization using pre-computed mean/std statistics

**ERA5DataLoader** (`era5_dataloader.py`):
- PyTorch Lightning DataModule interface
- Configurable batch size and worker processes

## Configuration

### Arguments (`baguan/utils/arguments.py`)

Default configuration:

```python
# Model
img_size = [721, 1440]
patch_size = 8
embed_dim = 1536
num_heads = 24
depth = 48

# Training
lr = 7e-4
betas = [0.9, 0.95]
weight_decay = 0.1
batch_size = 1

# Distributed
num_nodes = 2
devices = 8
precision = '16-mixed'
```

### DeepSpeed Config (`configs/ds_config.json`)

ZeRO Stage 2 optimization for distributed training.

## Installation

```bash
pip install -e .
```

Requirements:
- PyTorch
- PyTorch Lightning
- einops
- timm
- numpy
- torchvision

## Project Structure

```
baguan/
├── models/
│   ├── baguan_v2.py          # Main model architecture
│   ├── baguan_v1.py          # MAE-ClimaX variant
│   ├── fuxi.py               # Fuxi architecture
│   └── modules/              # Building blocks
│       ├── swin_transformer_v2_cr.py
│       ├── weather_embedding.py
│       └── ...
├── modules/
│   └── modules.py            # Lightning training module
├── datasets/
│   ├── era5_dataset.py       # ERA5 data loading
│   └── era5_dataloader.py    # DataModule
├── utils/
│   ├── arguments.py          # Configuration classes
│   ├── metrics.py            # Latitude-weighted metrics
│   └── lr_scheduler.py       # Custom schedulers
├── train.py                  # Main training script
└── train_v1.py               # V1 training script
```

## Data Format

Expected data structure:
```
root/
├── train/
│   └── {year}_{shard}/
│       └── {year}_{shard}_{idx}.npy
├── val/
├── test/
├── constant/
│   └── {var}.npy
├── normalize_mean.npz
└── normalize_std.npz
```

## License

MIT License
