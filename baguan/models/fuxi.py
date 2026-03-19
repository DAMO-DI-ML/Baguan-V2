import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
# from timm.layers.helpers import to_2tuple
# from timm.models.swin_transformer import SwinTransformerStage
from baguan.models.modules import BasicLayerV2, to_2tuple, SwinTransformerV2CrStage

from typing import Sequence
from einops import rearrange


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(torch.bfloat16)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class Fuxi(nn.Module):
    """
    Args:
        img_size (Sequence[int], optional): T, Lat, Lon.
        patch_size (Sequence[int], optional): T, Lat, Lon.
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        embed_dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """
    def __init__(self, img_size=(2, 720, 1440), patch_size=(2, 6, 6), in_chans=71, in_const=6, out_chans=71,
                 embed_dim=1536, num_groups=32, num_heads=16, window_size=10, depth=32, residual=True):
        super().__init__()

        self.single = False

        self.patch_size, self.embed_dim = patch_size, embed_dim
        self.in_chans, self.out_chans = in_chans, out_chans

        input_resolution = int(img_size[1] / patch_size[1]), int(img_size[2] / patch_size[2])

        # params of swin
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        self.input_resolution = list(input_resolution)

        self.hour_embedding = nn.Embedding(30, embed_dim)
        # self.day_embedding = TimestepEmbedder(embed_dim)
        self.day_embedding = nn.Embedding(400, embed_dim)
        self.absolute_pos_embed = nn.Parameter(
            torch.randn(1, self.input_resolution[0] * self.input_resolution[1], embed_dim) * .02, requires_grad=True
        )

        if self.single:
            self.patch_embedding = CubeEmbedding(
                (1, img_size[1], img_size[2]), (1, patch_size[1], patch_size[2]), in_chans + in_const, embed_dim
            )
        else:
            self.patch_embedding = CubeEmbedding(
                (2, img_size[1], img_size[2]), (2, patch_size[1], patch_size[2]), in_chans + in_const, embed_dim
            )
        
        img_window_ratio = 72
        dpr = [x.item() for x in torch.linspace(0, 0.2, depth)]
        self.layers = SwinTransformerV2CrStage(
            embed_dim=embed_dim,
            depth=depth,
            downscale=False,
            feat_size=(
                img_size[1] // patch_size[1],
                img_size[2] // patch_size[2],
            ),
            num_heads=num_heads,
            window_size=tuple([img_size[1] // img_window_ratio, img_size[2] // img_window_ratio]),
            drop_path=dpr,
            rel_pos=False,
            grad_checkpointing=True,
        )

        # self.head = FinalLayer(embed_dim, patch_size[1], out_chans)
        self.head = nn.Linear(embed_dim, patch_size[1] * patch_size[1] * out_chans)

        # self.layers._init_respostnorm()

    def forward(self, x: torch.Tensor, const: torch.Tensor, hour, date):
        B, _, _, Lat, Lon = x.shape
        _, patch_lat, patch_lon = self.patch_size
        shortcut = x[:, :, -1]

        if self.single:
            const = const.reshape(B, -1, 1, Lat, Lon)
            tokens = self.patch_embedding(torch.cat([x[:, :, [-1]], const], dim=1)).squeeze(2)
        else:
            const = const.reshape(B, -1, 1, Lat, Lon).repeat(1, 1, 2, 1, 1)
            tokens = self.patch_embedding(torch.cat([x, const], dim=1)).squeeze(2)
        
        _, _, H, W = tokens.shape
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        tokens = tokens + self.absolute_pos_embed + self.hour_embedding(hour) + self.day_embedding(date)

        tokens = self.layers(tokens) # b l d
        
        # day_temb = self.day_embedding(date[:, 0])
        # preds = self.head(tokens, day_temb)  # b l d
        preds = self.head(tokens)

        preds = rearrange(preds, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=H, c=self.out_chans, p1=self.patch_size[1])

        return preds + shortcut
