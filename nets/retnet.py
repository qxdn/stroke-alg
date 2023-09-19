import torch
import torch.nn as nn
from typing import Union, Sequence

from layers import PatchEmbedding, RetentionBlock


class RetNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        ffn_size: int = 3072,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        spatial_dims: int = 3,
        double_v_dim: bool = False,
    ) -> None:
        super(RetNet, self).__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, 
            img_size=img_size, 
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=spatial_dims,
        )

        self.blocks = nn.ModuleList([
            RetentionBlock(ffn_size, hidden_size, num_heads, double_v_dim) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x,hidden_states_out
