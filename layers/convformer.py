import torch.nn as nn
from monai.networks.blocks import SABlock, Convolution
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_norm_layer, get_act_layer, get_dropout_layer
from .activation import StarReLU

import einops


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        norm_name="instance",
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.norm = get_norm_layer(
            norm_name, channels=in_channels, spatial_dims=spatial_dims
        )
        self.conv = Conv[Conv.CONV, spatial_dims](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_name="starrelu",
        act2_name="identity",
        bias=False,
        kernel_size=7,
        padding=3,
        spatial_dims: int = 3,
    ):
        super().__init__()
        mid_channels = int(expansion_ratio * dim)
        self.pwconv1 = Conv[Conv.CONV, spatial_dims](
            dim, mid_channels, kernel_size=1, bias=bias
        )
        self.act1 = get_act_layer(act1_name)
        self.dwconv = Conv[Conv.CONV, spatial_dims](
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=mid_channels,
            bias=bias,
        )  # depthwise conv
        self.act2 = get_act_layer(act2_name)
        self.pwconv2 = Conv[Conv.CONV, spatial_dims](
            mid_channels, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class ConvFormerBlock(nn.Module):
    """
    metaformer version
    """

    def __init__(
        self,
        in_channels: int,
        norm_name: str = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        super(ConvFormerBlock, self).__init__()

        self.norm1 = get_norm_layer(
            norm_name, spatial_dims=spatial_dims, channels=in_channels
        )
        self.dwconv = SepConv(in_channels, spatial_dims=spatial_dims)
        self.dropout1 = get_dropout_layer(
            ("dropout", {"p": dropout_rate}), dropout_dim=spatial_dims
        )

        self.norm2 = get_norm_layer(
            norm_name, spatial_dims=spatial_dims, channels=in_channels
        )
        self.mlp = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](in_channels, 4 * in_channels, kernel_size=1),
            StarReLU(),
            get_dropout_layer(
                ("dropout", {"p": dropout_rate}), dropout_dim=spatial_dims
            ),
            Conv[Conv.CONV, spatial_dims](4 * in_channels, in_channels, kernel_size=1),
            get_dropout_layer(
                ("dropout", {"p": dropout_rate}), dropout_dim=spatial_dims
            ),
        )
        self.dropout2 = get_dropout_layer(
            ("dropout", {"p": dropout_rate}), dropout_dim=spatial_dims
        )

    def forward(self, x):
        x = x + self.dropout1(self.dwconv(self.norm1(x)))
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_dim: int = 32,
        num_heads=None,
        norm_name: str = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        super(TransformerBlock, self).__init__()

        self.spatial_dims = spatial_dims

        self.norm1 = get_norm_layer(norm_name, channels=in_channels)
        num_heads = num_heads if num_heads else in_channels // head_dim
        if num_heads == 0:
            num_heads = 1
        self.attn = SABlock(in_channels, num_heads, dropout_rate=dropout_rate)
        self.dropout1 = get_dropout_layer(("dropout", {"p": dropout_rate}))

        self.norm2 = get_norm_layer(norm_name, channels=in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            StarReLU(),
            get_dropout_layer(("dropout", {"p": dropout_rate})),
            nn.Linear(4 * in_channels, in_channels),
            get_dropout_layer(("dropout", {"p": dropout_rate})),
        )
        self.dropout2 = get_dropout_layer(("dropout", {"p": dropout_rate}))

    def forward(self, x):
        if self.spatial_dims == 3:
            B, C, H, W, D = x.shape
            x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        else:
            B, C, H, W = x.shape
            x = einops.rearrange(x, "b c h w -> b (h w) c")

        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.mlp(self.norm2(x)))

        if self.spatial_dims == 3:
            x = einops.rearrange(x, "b (h w d) c -> b c h w d", h=H, w=W, d=D)
        else:
            x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


class MetaFormerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        token_mixer_name: str = "convformer",
        norm_name: str = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super(MetaFormerStage, self).__init__()

        assert token_mixer_name in [
            "convformer",
            "transformer",
        ], "token mixer must be convformer or transformer"

        if token_mixer_name == "convformer":
            token_mixer = ConvFormerBlock
        else:
            token_mixer = TransformerBlock

        self.downsample = (
            nn.Identity()
            if in_channels == out_channels
            else Downsampling(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_name=norm_name,
            )
        )

        self.blocks = nn.ModuleList(
            [
                token_mixer(
                    out_channels,
                    norm_name=norm_name,
                    dropout_rate=dropout_rate,
                    spatial_dims=spatial_dims,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x
