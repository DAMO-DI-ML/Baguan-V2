from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from torch.utils.checkpoint import checkpoint
from timm.layers import DropPath, Mlp


ALL_VARIABLES = [
    # level variables
    "z_50", "q_50", "t_50", "u_50", "v_50", "z_100", "q_100", "t_100", "u_100", "v_100", 
    "z_150", "q_150", "t_150", "u_150", "v_150", "z_200", "q_200", "t_200", "u_200", "v_200", 
    "z_250", "q_250", "t_250", "u_250", "v_250", "z_300", "q_300", "t_300", "u_300", "v_300", 
    "z_400", "q_400", "t_400", "u_400", "v_400", "z_500", "q_500", "t_500", "u_500", "v_500", 
    "z_600", "q_600", "t_600", "u_600", "v_600", "z_700", "q_700", "t_700", "u_700", "v_700", 
    "z_850", "q_850", "t_850", "u_850", "v_850", "z_925", "q_925", "t_925", "u_925", "v_925", 
    "z_1000", "q_1000", "t_1000", "u_1000", "v_1000", 

    # surface variables
    "u10", "v10", "t2m", "msl", "u100", "v100", 'd2m', 'sp', 'sst', 'tcc', 'lcc',
    'tcw', 'tcwv', 'avg_sdswrf', 'avg_sdirswrf', 'tp1h', 'tp6h',

    # constant variables
    'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography', 
    'geopotential_at_surface', 'lake_cover', 'land_sea_mask', 'soil_type',

    # buffer variables
    'buffer_1', 'buffer_2', 'buffer_3', 'buffer_4', 'buffer_5', 'buffer_6', 'buffer_7', 
    'buffer_8', 'buffer_9', 'buffer_10', 'buffer_11', 'buffer_12', 'buffer_13', 'buffer_14', 
    'buffer_15', 'buffer_16', 'buffer_17', 'buffer_18', 'buffer_19', 'buffer_20', 'buffer_21', 
    'buffer_22', 'buffer_23', 'buffer_24', 'buffer_25', 'buffer_26', 'buffer_27', 'buffer_28', 
    'buffer_29', 'buffer_30', 'buffer_31', 'buffer_32', 'buffer_33', 'buffer_34', 'buffer_35', 
    'buffer_36', 'buffer_37', 'buffer_38', 'buffer_39', 'buffer_40',
]

class WeatherEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
        hidden_dim=128,
        variables=ALL_VARIABLES,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables
        self.hidden_dim = hidden_dim

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, hidden_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.channel_embed, self.channel_map = self.create_var_embedding(hidden_dim)

        self.channel_query = nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)
            
    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        # print(var_query.shape, self.channel_query.shape)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)

        # variable aggregation
        x = checkpoint(self.aggregate_variables, x, use_reentrant=False)
        # x = self.aggregate_variables(x)  # B, L, D
        x = self.proj(self.norm(x))

        return self.norm(x)


class HieraWeatherEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
        variables=ALL_VARIABLES,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables
        self.n_group = 12
        self.latent_ratio = 12
        self.hidden_dim = embed_dim // self.latent_ratio

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, self.hidden_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.channel_embed, self.channel_map = self.create_var_embedding(self.hidden_dim)

        self.channel_query = nn.Parameter(torch.rand(1, self.n_group, self.hidden_dim) * 0.2, requires_grad=True) # u v t q z sur
        self.channel_agg = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, num_heads // self.latent_ratio, batch_first=True)
        for i in range(self.n_group)])

        self.ensemble_query = nn.Parameter(torch.rand(1, self.latent_ratio, self.hidden_dim) * 0.2, requires_grad=True)
        self.ensemble_agg = nn.MultiheadAttention(self.hidden_dim, num_heads // self.latent_ratio, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)
            
    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor, group_ids):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        out_lst = []
        for i, g in enumerate([
            'u1', 'v1', 't1', 'q1', 'z1', 
            'u2', 'v2', 't2', 'q2', 'z2', 
            'surf', 'const'
        ]):
            kv = x[:, group_ids[g]]
            query = self.channel_query[:, [i]].repeat_interleave(kv.shape[0], dim=0)
            out, _ = self.channel_agg[i](query, kv, kv)  # BxL, 1, D
            out_lst.append(out)
        out = torch.concat(out_lst, dim=1) # BxL, 6, D

        ensemble_query = self.ensemble_query.repeat_interleave(out.shape[0], dim=0) # BxL, 4, D
        res, _ = self.ensemble_agg(ensemble_query, out, out) # BxL, 4, D
        res = res.flatten(1, 2) # BxL, 4D

        res = res.unflatten(dim=0, sizes=(b, l))  # B, L, 4D
        return res

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        
        group_ids = {
            'u1': [], 'v1': [], 'z1': [], 't1': [], 'q1': [], 
            'u2': [], 'v2': [], 'z2': [], 't2': [], 'q2': [], 
            'surf': [], 'const': []
        }
        for i, v in enumerate(variables):
            if v[1] == '_':
                if int(v[2:]) <= 400:
                    group_ids[f'{v[0]}1'].append(i)
                if int(v[2:]) <= 600:
                    group_ids[f'{v[0]}2'].append(i)
            else:
                if v in [
                    'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography', 
                    'geopotential_at_surface', 'lake_cover', 'land_sea_mask', 'soil_type'
                ]:
                    group_ids['const'].append(i)
                else:
                    group_ids['surf'].append(i)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)

        # variable aggregation
        # x = checkpoint(self.aggregate_variables, x, use_reentrant=False)
        x = self.aggregate_variables(x, group_ids)  # B, L, D

        return self.norm(x)