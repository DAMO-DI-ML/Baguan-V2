import torch
import torch.nn as nn

from baguan.models.modules import (
    SwinTransformerV2Cr, 
    SwinTransformerV2CrStage, 
    PatchEmbed, 
    bchw_to_bhwc,
    WeatherEmbedding,
    HieraWeatherEmbedding,
    ALL_VARIABLES,
)
from timm.layers import to_2tuple
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class BaguanV2(nn.Module):
    def __init__(self, checkpoint_stages=True):
        super(BaguanV2, self).__init__()

        img_size=(720, 1440)
        patch_size=6
        depths=(24,)
        num_heads=(16,)
        out_chans=len(ALL_VARIABLES)
        embed_dim=1536
        img_window_ratio=72
        drop_path_rate=0.2
        full_pos_embed=True
        rel_pos=False
        mlp_ratio=4.0
        checkpoint_stages=checkpoint_stages
        residual=True
        img_size = to_2tuple(img_size)
        window_size = tuple([s // img_window_ratio for s in img_size])

        self.weather_embedding = HieraWeatherEmbedding(
            img_size=img_size, 
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads[0]
        )
        self.patch_grid_size = [img_size[0] // patch_size, img_size[1] // patch_size]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [SwinTransformerV2CrStage(
                embed_dim=in_dim,
                depth=depth,
                downscale=False,
                feat_size=(
                    self.patch_grid_size[0] // in_scale,
                    self.patch_grid_size[1] // in_scale
                ),
                num_heads=num_heads,
                window_size=window_size,
                drop_path=dpr[stage_idx],
                extra_norm_stage=False or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                rel_pos=rel_pos,
                grad_checkpointing=checkpoint_stages,
            )]

        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(embed_dim, out_chans*patch_size*patch_size, bias=False)
        if full_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, self.patch_grid_size[0], self.patch_grid_size[1]) * .02)

        self.date_embedding = nn.Embedding(400, embed_dim)
        self.hour_embedding = nn.Embedding(30, embed_dim)

        self.patch_size = patch_size
        self.out_chans = out_chans
        self.img_size = img_size
        self.full_pos_embed = full_pos_embed

    def forward(self, x, hour, date, in_variables=None, out_variables=None):
        skip = x # 不包含tp
        
        x = self.weather_embedding(x, in_variables)
        x = rearrange(x, "b (h w) c->b c h w", h=self.patch_grid_size[0])

        if self.full_pos_embed:
            x = x + self.pos_embed
        x = x + \
            self.date_embedding(date[:, 0]).unsqueeze(dim=-1).unsqueeze(dim=-1) + \
                self.hour_embedding(hour[:, 0]).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.stages(x)

        B, _, h, w = x.shape
        x = bchw_to_bhwc(x)
        x = self.head(x)

        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_chans, self.img_size[0], self.img_size[1]))

        ids = self.weather_embedding.get_var_ids(tuple(out_variables), x.device)
        x = x[:, ids] # 包含tp

        for j, outv in enumerate(out_variables):
            for i, inv in enumerate(in_variables):
                if inv == outv:
                    x[:, j] = x[:, j] + skip[:, i]

        return x

if __name__ == '__main__':
    model = BaguanV2().cuda()
    hour = torch.Tensor([[1]]).long().cuda()
    date = torch.Tensor([[1]]).long().cuda()
    x = torch.rand(1, 77, 720, 1440).cuda()
    out = model(x, hour, date)
    print(out.shape)