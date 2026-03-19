# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baguan.models.modules.vision_transformer import Block, PatchEmbed, trunc_normal_
from baguan.models.modules.cross_block import CrossBlock
from baguan.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    PositionalEncoding3D,
)
from baguan.utils.timestepembedder import TimestepEmbedder, FinalLayer

from einops import repeat, rearrange


DEFAULT_VARS = [
    "land_sea_mask", "geopotential_at_surface", "angle_of_sub_gridscale_orography", "anisotropy_of_sub_gridscale_orography", "lake_cover", "soil_type",
    "z_50", "q_50", "t_50", "u_50", "v_50", "z_100", "q_100", "t_100", "u_100", "v_100", 
    "z_150", "q_150", "t_150", "u_150", "v_150", "z_200", "q_200", "t_200", "u_200", "v_200", 
    "z_250", "q_250", "t_250", "u_250", "v_250", "z_300", "q_300", "t_300", "u_300", "v_300", 
    "z_400", "q_400", "t_400", "u_400", "v_400", "z_500", "q_500", "t_500", "u_500", "v_500", 
    "z_600", "q_600", "t_600", "u_600", "v_600", "z_700", "q_700", "t_700", "u_700", "v_700", 
    "z_850", "q_850", "t_850", "u_850", "v_850", "z_925", "q_925", "t_925", "u_925", "v_925", 
    "z_1000", "q_1000", "t_1000", "u_1000", "v_1000", "u10", "v10", "t2m", "msl"
]


class MAEClimaX(nn.Module):
    def __init__(
        self,
        default_vars=DEFAULT_VARS,
        img_size=[721, 1440],
        patch_size=8,
        embed_dim=2048,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        dec_embed_dim=1024,
        dec_depth=4,
        dec_num_heads=8,
        pad=7,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        img_size = [img_size[0] + pad, img_size[1]]
        self.embed_dim = embed_dim
        self.pad = pad
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )

        self.num_patches = int((img_size[0] / patch_size) * (img_size[1] / patch_size))

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.pos_embed3d = PositionalEncoding3D(embed_dim)
        self.total_height = 1000 // 25 + 1
        self.height_map = {}
        for i in range(self.total_height - 1):
            self.height_map[str(i * 25)] = i + 1
        self.height = []
        for var in default_vars:
            try:
                self.height.append(self.height_map[var.split('_')[-1]])
            except:
                self.height.append(0)
        self.pos_3d_tokens = nn.Parameter(torch.zeros(
            1, img_size[0] // patch_size, img_size[1] // patch_size, self.total_height, self.embed_dim
        ))

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dec_embed_dim), requires_grad=True)
        
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.dec_lead_time_embed = nn.Linear(1, dec_embed_dim)

        self.enc_t_embedder = TimestepEmbedder(embed_dim)
        self.dec_t_embedder = TimestepEmbedder(dec_embed_dim)
        self.dec_fcst_t_embedder = TimestepEmbedder(dec_embed_dim)
        
        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # self.enc_to_dec = nn.Linear(embed_dim, dec_embed_dim) if embed_dim != dec_embed_dim else nn.Identity()
        self.new_enc_to_dec = nn.Linear(embed_dim, dec_embed_dim) if embed_dim != dec_embed_dim else nn.Identity()

        dec_dpr = [x.item() for x in torch.linspace(0, drop_path, dec_depth)]  # stochastic depth decay rule
        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         CrossBlock(
        #             dec_embed_dim,
        #             dec_num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             drop_path=dec_dpr[i],
        #             norm_layer=nn.LayerNorm,
        #             drop=drop_rate,
        #         )
        #         for i in range(dec_depth)
        #     ]
        # )
        self.fcst_decoder_blocks = nn.ModuleList(
            [
                CrossBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dec_dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(dec_depth)
            ]
        )

        self.mask_token = nn.Parameter(torch.randn(dec_embed_dim))
        self.fcst_to_pixels = FinalLayer(dec_embed_dim, patch_size, len(self.default_vars))

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        # print(f"self.pos_embed.shape = {self.pos_embed.shape}, {int(self.img_size[0] / self.patch_size)}, {int(self.img_size[1] / self.patch_size),}")
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.dec_pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.enc_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.enc_t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.dec_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.dec_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.encoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.fcst_decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.fcst_to_pixels.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fcst_to_pixels.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.fcst_to_pixels.linear.weight, 0)
        nn.init.constant_(self.fcst_to_pixels.linear.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]
        imgs = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=h, w=w, c=c, p1=p, p2=p)
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x
    
    def forward_encoder(self, x, lead_times, variables, mask_ratio):
        patches = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1 = self.patch_size, p2 = self.patch_size) # B V L P**2
        
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            token_emb = self.token_embeds[id](x[:, i : i + 1]) # b (h w) d
            pos_3d = self.pos_embed3d(self.pos_3d_tokens)[:, :, :, self.height[id]] # b h w d
            pos_3d = pos_3d.flatten(1, 2)
            token_emb = token_emb + pos_3d
            embeds.append(token_emb)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        tokens = self.aggregate_variables(x)  # B, L, D

        sp_pos_embed = None
        tokens = tokens + self.pos_embed

        # add time embedding
        tokens = tokens + self.lead_time_embed(lead_times.unsqueeze(-1)).unsqueeze(1)

        batch, nvar = x.shape[0], x.shape[1]
        num_masked = int(mask_ratio * self.num_patches)
        rand_indices = torch.rand(batch, self.num_patches, device = x.device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device = x.device)[:, None]

        if mask_ratio > 0:
            tokens = tokens[batch_range, unmasked_indices]
            # TODO: 不优雅
            patches = rearrange(patches, 'b c l d -> b l c d')
            masked_patches = patches[batch_range, masked_indices]
            patches = rearrange(patches, 'b l c d -> b c l d')
            masked_patches = rearrange(masked_patches, 'b l c d -> b c l d')
        else:
            masked_patches = None

        for i, blk in enumerate(self.encoder_blocks):
            time_embedding = self.enc_t_embedder(lead_times)
            tokens = blk(tokens, c=time_embedding)
        tokens = self.norm(tokens)

        return tokens, masked_patches, var_ids, num_masked, batch_range, masked_indices, unmasked_indices, sp_pos_embed, patches

    def forward(self, x, lead_times, variables, out_variables):
        x = F.pad(x, (0, 0, self.pad, 0), mode='replicate')

        if isinstance(variables, list):
            variables = tuple(variables)

        batch = x.shape[0]
        tokens_0, masked_patches, var_ids, num_masked, batch_range, masked_indices, unmasked_indices, sp_pos_embed, _ = \
            self.forward_encoder(x, lead_times, variables, 0)

        decoder_tokens_0 = self.new_enc_to_dec(tokens_0)

        dec_pos_embed = repeat(self.dec_pos_embed, 't n d -> (b t) n d', b = batch)
        mask_tokens = self.mask_token + dec_pos_embed

        # add time embedding
        dec_lead_time_emb = self.dec_lead_time_embed((lead_times - lead_times).unsqueeze(-1))  # B, D
        dec_lead_time_emb = dec_lead_time_emb.unsqueeze(1)
        dec_lead_time_emb = repeat(dec_lead_time_emb, 'b n d -> b (t n) d', t = dec_pos_embed.shape[1])
        mask_tokens = mask_tokens + dec_lead_time_emb

        """
        step3: Decoder: Forward
        """
        decoder_tokens_1 = mask_tokens
        for blk in self.fcst_decoder_blocks:
            dec_time_embedding = self.dec_t_embedder(lead_times)
            decoder_tokens_1 = blk(decoder_tokens_0, decoder_tokens_1, dec_time_embedding)
        # decoder_tokens_1 = decoder_tokens_0

        dec_time_embedding = self.dec_t_embedder(lead_times)
        preds = self.fcst_to_pixels(decoder_tokens_1, dec_time_embedding)  # B, L, V*p*p
        
        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)

        preds = preds[:, out_var_ids, self.pad:]
        
        return preds
