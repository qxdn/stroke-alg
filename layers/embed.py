from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from typing import Sequence, Union

class PatchEmbedding(PatchEmbeddingBlock):

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__(in_channels, img_size, patch_size, hidden_size, num_heads, pos_embed, dropout_rate, spatial_dims)

    
    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x
        embeddings = self.dropout(embeddings)
        return embeddings