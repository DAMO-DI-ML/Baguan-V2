import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from einops import rearrange
from torch.utils.checkpoint import checkpoint

import numpy as np
from sklearn.neighbors import KDTree


class SphericalEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels=1536, 
        out_channels=1536,
        num_heads=16,
        img_size=(720, 1440), 
        patch_size=6,
        topk=5,
    ):
        super(SphericalEmbedding, self).__init__()

        self.topk = topk
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_out_nodes = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.n_inp_nodes = self.n_out_nodes

        self.out_node_embedding = nn.Parameter(torch.randn(1, self.n_out_nodes, in_channels), requires_grad=True)
        self.gat = GATConv(
            in_channels, 
            out_channels // num_heads, 
            heads=num_heads, 
            concat=True, 
            # add_self_loops=False
        )
        self.norm = nn.LayerNorm(out_channels)

        fibonacci_points = fibonacci_sphere(self.n_out_nodes)
        fibonacci_geographic_points = cartesian_to_geographic(fibonacci_points)
        fibonacci_cartesian_points = np.array([geographic_to_cartesian(lat, lon) for lat, lon in fibonacci_geographic_points])
        tree = KDTree(fibonacci_cartesian_points)

        lat_lon_grid = create_lat_lon_grid(self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size)
        grid_cartesian_points = np.array([geographic_to_cartesian(lat, lon) for lat, lon in lat_lon_grid])
        dist, ind = tree.query(grid_cartesian_points, k=self.topk)

        self.edge = self.build_edge(ind)

    def build_edge(self, ind):
        edge_lst = []
        for i, nearest_id in enumerate(ind):
            for j in nearest_id:
                edge_lst.append([i, j + self.n_inp_nodes])
        
        edge = torch.Tensor(edge_lst).t().long()
        return edge

    def forward(self, x):
        """
            x: [B, C, H, W]

            return: [B, N (n_out_nodes), D (out_channels)]
                        -> [B, D, h, w]
        """
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = torch.cat([x, self.out_node_embedding], dim=1)

        out = self.gat(x[0], self.edge.to(x.device)).unsqueeze(dim=0)[:, -self.n_out_nodes:]
        out = self.norm(out)
        out = rearrange(out, 'b (h w) d -> b d h w', h=self.img_size[0] // self.patch_size)

        return out


class ReverseSphericalEmbedding(SphericalEmbedding):
    def build_edge(self, ind):
        edge_lst = []
        for i, nearest_id in enumerate(ind):
            for j in nearest_id:
                edge_lst.append([j, i + self.n_inp_nodes])
        
        edge = torch.Tensor(edge_lst).t().long()
        return edge


def fibonacci_sphere(N):
    points = []
    phi = (1 + np.sqrt(5)) / 2  # 黄金比例
    
    for i in range(N):
        theta = np.arccos(1 - 2 * (i + 0.5) / N)
        phi_i = 2 * np.pi * phi * i
        
        x = np.cos(phi_i) * np.sin(theta)
        y = np.sin(phi_i) * np.sin(theta)
        z = np.cos(theta)
        
        points.append((x, y, z))
    
    return np.array(points)


def cartesian_to_geographic(points):
    geographic_points = []
    
    for x, y, z in points:
        latitude = np.arcsin(z) * 180 / np.pi
        longitude = np.arctan2(y, x) * 180 / np.pi
        geographic_points.append((latitude, longitude))
    
    return np.array(geographic_points)


def create_lat_lon_grid(lat_steps, lon_steps):
    latitudes = np.linspace(90, -90, lat_steps)
    longitudes = np.linspace(0, 360, lon_steps, endpoint=False)
    grid_points = []
    
    for lat in latitudes:
        for lon in longitudes:
            grid_points.append((lat, lon))
    
    return np.array(grid_points)


def geographic_to_cartesian(lat, lon):
    # 经纬度转换为笛卡尔坐标
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return (x, y, z)

if __name__ == '__main__':
    model = SphericalEmbedding().cuda()
    x = torch.rand(1, 1536, 720 // 6, 1440 // 6).cuda()
    out = model(x)
    print(out.shape)
    reverse_model = ReverseSphericalEmbedding().cuda()
    reverse = reverse_model(x)
    print(reverse.shape)