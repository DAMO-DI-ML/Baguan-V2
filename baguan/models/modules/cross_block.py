import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import deepspeed as ds
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from torch.nn.attention import SDPBackend, sdpa_kernel

from timm.models.vision_transformer import LayerScale, PatchEmbed, Mlp, DropPath
from baguan.models.modules.vision_transformer import Attention

from baguan.utils.timestepembedder import modulate
from torch.utils.checkpoint import checkpoint


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rope_theta=100.0, img_size=[32, 64], patch_size=2,):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def cal_attn(self, inputs):
        cnt = inputs.shape[0]
        xq, xk, xv = inputs[:cnt//3], inputs[cnt//3:cnt//3*2], inputs[cnt//3*2:]
        B, N, C = xq.shape

        q = self.wq(xq).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(xk).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(xv).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.attn_drop_rate, 
            ).transpose(1, 2).reshape(B, N, C)
        
        return x

    def forward(self, xq, xk, xv):
        inputs = torch.cat([xq, xk, xv], dim=0)

        x = checkpoint(self.cal_attn, inputs, use_reentrant=False)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm01 = norm_layer(dim)
        self.norm02 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls0 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x1, x2, c):
        temp_res = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temp_res.chunk(6, dim=1)

        x1_norm = self.norm01(x1)
        x2_norm = self.norm02(x2)

        x = self.drop_path0(self.ls0(self.cross_attn(x2_norm, x1_norm, x1_norm)))
        x = x1 + x

        temp_res = checkpoint(self.attn, modulate(self.norm1(x), shift_msa, scale_msa), use_reentrant=False)
        # temp_res = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        _x_ = gate_msa.unsqueeze(1) * self.drop_path1(self.ls1(temp_res))
        x = x + _x_

        temp_res = checkpoint(self.mlp, modulate(self.norm2(x), shift_mlp, scale_mlp), use_reentrant=False)
        # temp_res = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        _x_ = gate_mlp.unsqueeze(1) * self.drop_path2(self.ls2(temp_res))
        x = x + _x_

        return x1
