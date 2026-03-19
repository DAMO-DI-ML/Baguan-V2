# Baguan-V2 八观气象大模型

基于 Swin Transformer V2 架构的全球天气预报深度学习框架。

## 项目简介

Baguan-V2（八观V2）是一个基于 PyTorch 的全球气象预报模型，采用 Swin Transformer V2 交叉分辨率注意力机制架构。该模型专为中期全球天气预报设计，使用 ERA5 再分析数据进行训练。

## 模型架构

### 核心模型 (`baguan/models/baguan_v2.py`)

主要架构组成：

- **HieraWeatherEmbedding**: 层次化气象变量嵌入层
- **SwinTransformerV2CrStage**: Swin Transformer V2 交叉分辨率阶段
- **多头注意力**: 16 个注意力头，1536 维嵌入维度
- **Patch 大小**: 在 720x1440 经纬度网格上使用 6x6 空间 Patch
- **网络深度**: 24 层 Transformer
- **窗口大小**: 基于图像窗口比例（72）的动态窗口大小

主要特性：
- 完整的位置嵌入用于空间信息编码
- 日期和时间嵌入用于时间编码
- 残差连接确保训练稳定
- 梯度检查点节省显存

### 备选模型

- **BaguanV1** (`baguan/models/baguan_v1.py`): 基于 MAE 的 ClimaX 架构，支持变量标记化
- **Fuxi** (`baguan/models/fuxi.py`): 3D 立方体嵌入与 Swin Transformer 阶段

## 训练流程

### 主训练脚本 (`baguan/train.py`)

训练流程基于 PyTorch Lightning，包含：

- **并行策略**: FSDP（完全分片数据并行）
- **日志记录**: Weights & Biases (wandb)
- **模型检查点**: Top-k 模型保存与损失监控
- **混合精度**: 16 位混合精度训练

```bash
python -m baguan.train
```

### 训练模块 (`baguan/modules/modules.py`)

`BaguanV2Module` 实现了：
- 自回归训练与多步 rollout
- 纬度加权 MAE 和 RMSE 评估指标
- AdamW 优化器（区分权重衰减参数）
- 线性预热余弦退火学习率调度器

### 数据管道 (`baguan/datasets/`)

**ERA5Datasets** (`era5_dataset.py`):
- 支持地表和高空变量
- 13 个气压层（50-1000 hPa）
- 5 个高空变量：位势高度(z)、比湿(q)、温度(t)、纬向风(u)、经向风(v)
- 16 个地表变量，包括降水（tp1h, tp6h）
- 6 个常量变量（地形、海陆掩膜等）
- 使用预计算的均值/标准差进行归一化

**ERA5DataLoader** (`era5_dataloader.py`):
- PyTorch Lightning DataModule 接口
- 可配置的批次大小和工作进程数

## 配置说明

### 参数配置 (`baguan/utils/arguments.py`)

默认配置：

```python
# 模型参数
img_size = [721, 1440]      # 图像尺寸（纬度x经度）
patch_size = 8              # Patch 大小
embed_dim = 1536            # 嵌入维度
num_heads = 24              # 注意力头数
depth = 48                  # 网络深度

# 训练参数
lr = 7e-4                   # 学习率
betas = [0.9, 0.95]         # Adam 优化器 beta 参数
weight_decay = 0.1          # 权重衰减
batch_size = 1              # 批次大小

# 分布式训练
num_nodes = 2               # 节点数
devices = 8                 # 每节点 GPU 数
precision = '16-mixed'      # 混合精度
```

### DeepSpeed 配置 (`configs/ds_config.json`)

ZeRO Stage 2 优化用于分布式训练。

## 安装方法

```bash
pip install -e .
```

依赖项：
- PyTorch
- PyTorch Lightning
- einops
- timm
- numpy
- torchvision

## 项目结构

```
baguan/
├── models/
│   ├── baguan_v2.py          # 主模型架构
│   ├── baguan_v1.py          # MAE-ClimaX 变体
│   ├── fuxi.py               # Fuxi 架构
│   └── modules/              # 基础组件
│       ├── swin_transformer_v2_cr.py
│       ├── weather_embedding.py
│       └── ...
├── modules/
│   └── modules.py            # Lightning 训练模块
├── datasets/
│   ├── era5_dataset.py       # ERA5 数据加载
│   └── era5_dataloader.py    # DataModule
├── utils/
│   ├── arguments.py          # 配置类
│   ├── metrics.py            # 纬度加权指标
│   └── lr_scheduler.py       # 自定义学习率调度器
├── train.py                  # 主训练脚本
└── train_v1.py               # V1 训练脚本
```

## 数据格式

预期的数据结构：
```
root/
├── train/                    # 训练数据
│   └── {year}_{shard}/       # 年份_分片
│       └── {year}_{shard}_{idx}.npy
├── val/                      # 验证数据
├── test/                     # 测试数据
├── constant/                 # 常量数据
│   └── {var}.npy            # 常量变量（如地形）
├── normalize_mean.npz        # 归一化均值
└── normalize_std.npz         # 归一化标准差
```

## 变量说明

### 高空变量（13个气压层）
- 位势高度 (z): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa
- 比湿 (q): 同上
- 温度 (t): 同上
- 纬向风 (u): 同上
- 经向风 (v): 同上

### 地表变量
- u10, v10: 10米风速
- t2m: 2米温度
- msl: 平均海平面气压
- u100, v100: 100米风速
- d2m: 2米露点温度
- sp: 地表气压
- tcc: 总云量
- lcc: 低云量
- avg_sdswrf: 平均地表向下短波辐射通量
- avg_sdirswrf: 平均地表直接短波辐射通量
- tcw: 总柱水
- tcwv: 总柱水汽
- tp1h: 1小时累计降水
- tp6h: 6小时累计降水
- sst: 海表温度

### 常量变量
- 亚网格地形坡度角
- 亚网格地形各向异性
- 地表位势高度
- 湖泊覆盖
- 海陆掩膜
- 土壤类型

## 许可证

MIT License
