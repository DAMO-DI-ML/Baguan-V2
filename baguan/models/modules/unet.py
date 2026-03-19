import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, stride=2,
            padding=1, padding_mode='zeros',
        )
        
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim, embed_dim, 
                kernel_size=3, stride=1,
                padding=1, padding_mode='zeros',
            ),
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim, embed_dim, 
                kernel_size=3, stride=1,
                padding=1, padding_mode='zeros',
            ),
        )
    
    def forward(self, x):
        x = self.conv(x)
        out = x + self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        out_embed_dim = embed_dim // 2
        self.conv2d = nn.Conv2d(embed_dim, out_embed_dim, kernel_size=1, stride=1)

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim, out_embed_dim, 
                kernel_size=3, stride=1,
                padding=1, padding_mode='zeros',
            ),
            nn.GroupNorm(num_groups=32, num_channels=out_embed_dim),
            nn.SiLU(),
            nn.Conv2d(
                out_embed_dim, out_embed_dim, 
                kernel_size=3, stride=1,
                padding=1, padding_mode='zeros',
            ),
        )

        self.convtrans = nn.ConvTranspose2d(
            out_embed_dim, out_embed_dim,
            kernel_size=2, stride=2,
        )
        
    def forward(self, x):
        # [b, n, 3072]
        x = self.conv2d(x) + self.block(x)
        return self.convtrans(x)


if __name__ == "__main__":
    x = torch.rand(1, 3072, 180, 90)
    # down = DownBlock(1536)
    # x = down(x)
    # print(x.shape)
    up = UpBlock(1536 * 2)
    y = up(x)
    print(y.shape)
