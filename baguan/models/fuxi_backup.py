import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.layers import PatchEmbed
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from baguan.models.modules import BasicLayer

from typing import Sequence
from einops import rearrange


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class DownBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, skip_scale=1):
        super().__init__()
        self.skip_scale = skip_scale
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        self.in_layer = nn.Sequential(
            nn.GroupNorm(num_groups, out_chans),
            # nn.Conv2d(out_chans, out_chans, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
        )

        self.out_layer_0 = nn.Sequential(
            nn.GroupNorm(num_groups, out_chans),
            # nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        self.out_layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_chans, 2 * out_chans, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, temb):
        B, _, h, w = x.shape

        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=-1)
        x = self.conv(x)
        shortcut = x

        x = self.in_layer(x)
        s = self.out_layer_0(x)
        x = modulate(x.permute(0, 2, 3, 1), shift, scale).permute(0, 3, 1, 2)
        x = self.out_layer_1(x)

        return (x + shortcut) / self.skip_scale


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, skip_scale=1):
        super().__init__()
        self.skip_scale = skip_scale
        self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1) if in_chans != out_chans else nn.Identity()

        self.in_layer = nn.Sequential(
            nn.GroupNorm(num_groups, in_chans),
            # nn.Conv2d(in_chans, in_chans, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
        )

        self.out_layer_0 = nn.Sequential(
            nn.GroupNorm(num_groups, out_chans),
            # nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        self.out_layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_chans, 2 * out_chans, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, temb):
        B = x.shape[0]

        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=-1)
        shortcut = self.shortcut_conv(x)

        x = self.in_layer(x)
        x = self.out_layer_0(x)
        x = modulate(x.permute(0, 2, 3, 1), shift, scale).permute(0, 3, 1, 2)
        x = self.out_layer_1(x)

        return (x + shortcut) / self.skip_scale


class SwinBlocks(nn.Module):
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0, depth)]
        self.layer_1 = SwinTransformerV2Stage(
            embed_dim, embed_dim, input_resolution, depth // 4, 
            num_heads, window_size, 
            # drop_path=dpr
        )
        self.layer_1.grad_checkpointing = True
        self.proj_ln_1 = nn.LayerNorm(embed_dim)
        self.layer_2 = SwinTransformerV2Stage(
            embed_dim, embed_dim, input_resolution, depth // 4, 
            num_heads, window_size, 
            # drop_path=dpr
        )
        self.layer_2.grad_checkpointing = True
        self.proj_ln_2 = nn.LayerNorm(embed_dim)
        self.layer_3 = SwinTransformerV2Stage(
            embed_dim, embed_dim, input_resolution, depth // 4, 
            num_heads, window_size, 
            # drop_path=dpr
        )
        self.layer_3.grad_checkpointing = True
        self.proj_ln_3 = nn.LayerNorm(embed_dim)
        self.layer_4 = SwinTransformerV2Stage(
            embed_dim, embed_dim, input_resolution, depth // 4, 
            num_heads, window_size, 
            # drop_path=dpr
        )
        self.layer_4.grad_checkpointing = True
        self.proj_ln_4 = nn.LayerNorm(embed_dim)

        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU()
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        out_1 = self.layer_1(x) # B Lat Lon C
        out_2 = self.layer_2(out_1) # B Lat Lon C
        out_3 = self.layer_3(out_2) # B Lat Lon C
        out_4 = self.layer_4(out_3) # B Lat Lon C
        out_1 = self.proj_ln_1(out_1)
        out_2 = self.proj_ln_2(out_2)
        out_3 = self.proj_ln_3(out_3)
        out_4 = self.proj_ln_4(out_4)
        out = torch.cat([out_1, out_2, out_3, out_4], dim=-1) # B Lat Lon 4C
        out = self.fpn(out).permute(0, 3, 1, 2) # B C Lat Lon

        return out


class UTransformer(nn.Module):
    """U-Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        num_groups (int | tuple[int]): number of groups to separate the channels into.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])
        self.layer = SwinBlocks(embed_dim, num_groups, input_resolution, num_heads, window_size, depth)
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x, temb):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.down(x, temb)

        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        out = self.layer(x)

        # crop
        out = out[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        out = torch.cat([shortcut, out], dim=1)  # B 2*C Lat Lon

        out = self.up(out, temb)
        return out


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
    def __init__(self, img_size=(2, 721, 1440), patch_size=(2, 4, 4), in_chans=75, out_chans=69,
                 embed_dim=1536, num_groups=32, num_heads=24, window_size=18, residual=True):
        super().__init__()
        self.in_chans, self.out_chans = in_chans, out_chans
        self.residual = residual

        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)
        # self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.patch_embedding = PatchEmbed(
            img_size=(img_size[1], img_size[2]), patch_size=patch_size[1], 
            in_chans=in_chans*2, embed_dim=embed_dim, flatten=False
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        self.u_transformer = UTransformer(embed_dim, num_groups, input_resolution, num_heads, window_size, depth=48)
        # self.fc = nn.Linear(embed_dim, out_chans * patch_size[1] * patch_size[2])
        self.conv_trans = nn.ConvTranspose2d(
            embed_dim,  embed_dim, kernel_size=2*2, stride=2
        )
        self.fc = nn.Linear(embed_dim, out_chans * patch_size[1] * patch_size[2])
        # if self.residual:
        #     self.scale = nn.Parameter(torch.zeros(1, out_chans, 1, 1), requires_grad=True)

        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size

        self.time_embedding = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, hour, date):
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2

        if self.residual:
            shortcut = x[:, :self.out_chans, 1]
        
        temb = self.get_time_embedding(hour, date) # B, D

        # x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.patch_embedding(x)
        x = self.patch_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.u_transformer(x, temb) # B C Lat Lon
        x = self.conv_trans(x).permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.fc(x)
        x = rearrange(x, 'b h w (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size[1], p2=self.patch_size[2])

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=False)

        if self.residual:
            # x = x * self.scale + shortcut
            x = x + shortcut

        return x

    def get_time_embedding(self, hour, date):

        temb = []

        for i in [-6, 0, 6]:
            temp_hour = hour.clone() + i
            temp_dates = date.clone()
            if temp_hour < 0:
                temp_hour += 24
                temp_dates -= 1
            elif temp_hour >= 24:
                temp_hour -=24
                temp_dates += 1
            
            temb.append(temp_hour / 24)
            temb.append(temp_dates / 366)
        
        temb = torch.cat(temb, dim=-1)
        temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=-1) # [b, 12]

        temb = temb.to(next(self.time_embedding.parameters()).dtype)
        temb = self.time_embedding(temb).unsqueeze(dim=1) # [b, 1, d]

        return temb