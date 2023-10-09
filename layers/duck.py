from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_norm_layer, get_act_layer


class WideScope(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        spatial_dims: int = 3,
    ):
        super(WideScope, self).__init__()

        self.conv1 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=1,
        )

        self.conv2 = Convolution(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=2,
        )

        self.conv3 = Convolution(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=3,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MidScope(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        spatial_dims: int = 3,
    ):
        super(MidScope, self).__init__()

        self.conv1 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=1,
        )

        self.conv2 = Convolution(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=2,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DuckResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        dilation: Union[Sequence[int], int] = 1,
        spatial_dims: int = 3,
    ) -> None:
        super(DuckResidualBlock, self).__init__()

        self.residual = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=dilation,
        )

        self.conv1 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=dilation,
        )

        self.conv2 = Convolution(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
            dilation=dilation,
        )

        self.norm = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.residual(identity)

        x += identity

        x = self.norm(x)

        return x


class SeparatedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int = 3,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        spatial_dims: int = 3,
    ) -> None:
        super(SeparatedConvBlock, self).__init__()

        self.conv1 = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=(size, 1, 1) if spatial_dims == 3 else (size, 1),
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
        )

        self.conv2 = Convolution(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=(1, size, 1) if spatial_dims == 3 else (1, size),
            adn_ordering=adn_ordering,
            norm=norm_name,
            act=act_name,
        )

        if spatial_dims == 3:
            self.conv3 = Convolution(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=(1, 1, size),
                adn_ordering=adn_ordering,
                norm=norm_name,
                act=act_name,
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class DuckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        spatial_dims: int = 3,
    ):
        super(DuckBlock, self).__init__()

        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=in_channels
        )

        self.ws = WideScope(
            in_channels=in_channels,
            out_channels=out_channels,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
            spatial_dims=spatial_dims,
        )

        self.ms = MidScope(
            in_channels=in_channels,
            out_channels=out_channels,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
            spatial_dims=spatial_dims,
        )

        self.res1 = DuckResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
            spatial_dims=spatial_dims,
        )

        self.res2 = nn.Sequential(
            *[
                DuckResidualBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    act_name=act_name,
                    norm_name=norm_name,
                    adn_ordering=adn_ordering,
                    spatial_dims=spatial_dims,
                )
                for i in range(2)
            ]
        )

        self.res3 = nn.Sequential(
            *[
                DuckResidualBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    act_name=act_name,
                    norm_name=norm_name,
                    adn_ordering=adn_ordering,
                    spatial_dims=spatial_dims,
                )
                for i in range(3)
            ]
        )

        self.sepconv = SeparatedConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            size=5,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
            spatial_dims=spatial_dims,
        )

        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )

    def forward(self, x_in):
        x_in = self.norm1(x_in)

        ws = self.ws(x_in)
        ms = self.ms(x_in)
        res1 = self.res1(x_in)
        res2 = self.res2(x_in)
        res3 = self.res3(x_in)
        sepconv = self.sepconv(x_in)

        x_final = ws + ms + res1 + res2 + res3 + sepconv

        x_final = self.norm2(x_final)

        return x_final
