from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.upsample import UpSample
from monai.networks.blocks.convolutions import Convolution
from monai.utils import ensure_tuple_rep

from layers import DuckBlock, DuckResidualBlock


class DuckNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        start_channels: int = 17,
        act_name: Optional[Union[Tuple, str]] = "RELU",
        norm_name: Optional[Union[Tuple, str]] = "INSTANCE",
        adn_ordering: str = "ADN",
        interp_mode: str = "nearest",
        align_corners: Optional[bool] = None,
        spatial_dims: int = 3,
    ):
        super(DuckNet, self).__init__()

        upsamle_size = ensure_tuple_rep(2, spatial_dims)

        self.down1 = Convolution(
            spatial_dims,
            in_channels,
            start_channels * 2,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.down2 = Convolution(
            spatial_dims,
            start_channels * 2,
            start_channels * 4,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.down3 = Convolution(
            spatial_dims,
            start_channels * 4,
            start_channels * 8,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.down4 = Convolution(
            spatial_dims,
            start_channels * 8,
            start_channels * 16,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.down5 = Convolution(
            spatial_dims,
            start_channels * 16,
            start_channels * 32,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.duck1 = DuckBlock(
            in_channels,
            start_channels,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )
        self.sub_down1 = Convolution(
            spatial_dims,
            start_channels,
            start_channels * 2,
            kernel_size=2,
            strides=2,
            padding=0
        )

        self.duck2 = DuckBlock(
            start_channels * 2,
            start_channels * 2,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.sub_down2 = Convolution(
            spatial_dims,
            start_channels * 2,
            start_channels * 4,
            kernel_size=2,
            strides=2,
            padding=0
        )
        self.duck3 = DuckBlock(
            start_channels * 4,
            start_channels * 4,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.sub_down3 = Convolution(
            spatial_dims,
            start_channels * 4,
            start_channels * 8,
            kernel_size=2,
            strides=2,
            padding=0
        )
        self.duck4 = DuckBlock(
            start_channels * 8,
            start_channels * 8,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.sub_down4 = Convolution(
            spatial_dims,
            start_channels * 8,
            start_channels * 16,
            kernel_size=2,
            strides=2,
            padding=0
        )
        self.duck5 = DuckBlock(
            start_channels * 16,
            start_channels * 16,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.sub_down5 = Convolution(
            spatial_dims,
            start_channels * 16,
            start_channels * 32,
            kernel_size=2,
            strides=2,
            padding=0
        )
        self.res1 = nn.Sequential(
            DuckResidualBlock(
                start_channels * 32,
                start_channels * 32,
                act_name=act_name,
                norm_name=norm_name,
                adn_ordering=adn_ordering,
                spatial_dims=spatial_dims,
            ),
            DuckResidualBlock(
                start_channels * 32,
                start_channels * 32,
                act_name=act_name,
                norm_name=norm_name,
                adn_ordering=adn_ordering,
                spatial_dims=spatial_dims,
            ),
        )
        self.res2 = nn.Sequential(
            DuckResidualBlock(
                start_channels * 32,
                start_channels * 16,
                act_name=act_name,
                norm_name=norm_name,
                adn_ordering=adn_ordering,
                spatial_dims=spatial_dims,
            ),
            DuckResidualBlock(
                start_channels * 16,
                start_channels * 16,
                act_name=act_name,
                norm_name=norm_name,
                adn_ordering=adn_ordering,
                spatial_dims=spatial_dims,
            ),
        )

        self.up1 = UpSample(
            spatial_dims=3,
            scale_factor=upsamle_size,
            mode="nontrainable",
            interp_mode=interp_mode,
            align_corners=align_corners
        )
        self.duck6 = DuckBlock(
            start_channels * 16,
            start_channels * 8,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.up2 = UpSample(
            spatial_dims=3,
            scale_factor=upsamle_size,
            mode="nontrainable",
            interp_mode=interp_mode,
            align_corners=align_corners
        )
        self.duck7 = DuckBlock(
            start_channels * 8,
            start_channels * 4,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.up3 = UpSample(
            spatial_dims=3,
            scale_factor=upsamle_size,
            mode="nontrainable",
            interp_mode=interp_mode,
            align_corners=align_corners
        )
        self.duck8 = DuckBlock(
            start_channels * 4,
            start_channels * 2,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )
        
        self.up4 = UpSample(
            spatial_dims=3,
            scale_factor=upsamle_size,
            mode="nontrainable",
            interp_mode=interp_mode,
            align_corners=align_corners
        )
        self.duck9 = DuckBlock(
            start_channels * 2,
            start_channels,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.up5 = UpSample(
            spatial_dims=3,
            scale_factor=upsamle_size,
            mode="nontrainable",
            interp_mode=interp_mode,
            align_corners=align_corners
        )
        self.duck10 = DuckBlock(
            start_channels,
            start_channels,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            adn_ordering=adn_ordering,
        )

        self.final_conv = Convolution(
            spatial_dims,
            start_channels,
            out_channels,
            kernel_size=1,
            act=act_name,
            norm=norm_name,
        )

    def forward(self, x):
        p1 = self.down1(x)
        p2 = self.down2(p1)
        p3 = self.down3(p2)
        p4 = self.down4(p3)
        p5 = self.down5(p4)

        t0 = self.duck1(x)
        l1i = self.sub_down1(t0)
        s1 = p1 + l1i
        t1 = self.duck2(s1)

        l2i = self.sub_down2(t1)
        s2 = p2 + l2i
        t2 = self.duck3(s2)

        l3i = self.sub_down3(t2)
        s3 = p3 + l3i
        t3 = self.duck4(s3)

        l4i = self.sub_down4(t3)
        s4 = p4 + l4i
        t4 = self.duck5(s4)

        l5i = self.sub_down5(t4)
        s5 = p5 + l5i
        t51 = self.res1(s5)
        t53 = self.res2(t51)

        l5o = self.up1(t53)
        c4 = l5o + t4
        q4 = self.duck6(c4)

        l4o = self.up2(q4)
        c3 = l4o + t3
        q3 = self.duck7(c3)

        l3o = self.up3(q3)
        c2 = l3o + t2
        q2 = self.duck8(c2)

        l2o = self.up4(q2)
        c1 = l2o + t1
        q1 = self.duck9(c1)

        l1o = self.up5(q1)
        c0 = l1o + t0
        z1 = self.duck10(c0)

        y = self.final_conv(z1)
        return y
