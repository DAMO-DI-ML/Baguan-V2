import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=[16, 16], in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
