import torch.nn as nn
import torch
from typing import Tuple, Union, Sequence, Optional
from layers import Stem, MetaFormerStage,MetaPolypConvFormerBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_norm_layer, get_act_layer
from monai.networks.blocks.dynunet_block import get_conv_layer


class UpSample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
        add: bool = False,
    ) -> None:
        super(UpSample, self).__init__()

        self.add = add

        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels + in_channels if not self.add else in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = UnetrBasicBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

    def forward(self, inp, skip):
        if self.add:
            out = inp + skip
        else:
            out = torch.cat((inp, skip), dim=1)
        out = self.transp_conv(out)
        out = self.conv_block(out)
        return out


class CAFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        token_mixers=("convformer", "convformer", "transformer", "transformer"),
        drop_path_rate=0.0,
        spatial_dims: int = 3,
    ) -> None:
        super(CAFormer, self).__init__()

        self.num_stages = len(depths)

        self.stem = Stem(
            in_channels=in_channels, out_channels=dims[0], spatial_dims=spatial_dims
        )

        stages = []
        prev_dim = dims[0]

        for i in range(self.num_stages):
            stages.append(
                MetaFormerStage(
                    prev_dim,
                    dims[i],
                    num_layers=depths[i],
                    token_mixer_name=token_mixers[i],
                    norm_name=norm_name,
                    dropout_rate=drop_path_rate,
                    spatial_dims=spatial_dims,
                )
            )
            prev_dim = dims[i]

        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)

        hidden_states = []
        for stage in self.stages:
            x = stage(x)
            hidden_states.append(x)

        return x, hidden_states


class CAFormerUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        drop_path_rate=0.0,
        res_block: bool = True,
        add: bool = False,
    ) -> None:
        super(CAFormerUnet, self).__init__()

        self.caformer = CAFormer(
            in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,
        )

        self.skip_encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.skip_encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.skip_encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.skip_encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder2 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder3 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder4 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0] // 4,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.final_conv = get_conv_layer(
            spatial_dims,
            dims[0] // 4,
            out_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act,
        )

    def forward(self, x):
        x, hidden_states = self.caformer(x)

        y = self.skip_encoder1(hidden_states[3])  # /32
        x = self.decoder1(x, y)  # /16

        y = self.skip_encoder2(hidden_states[2])  # /16
        x = self.decoder2(x, y)  # /8

        y = self.skip_encoder3(hidden_states[1])  # /8
        x = self.decoder3(x, y)  # /4

        y = self.skip_encoder4(hidden_states[0])  # /4
        x = self.decoder4(x, y)  # /1

        x = self.final_conv(x)

        return x


class SimpleUpSample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        act_name: Union[Tuple, str] = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        upsameple_only: bool = False
    ) -> None:
        super(SimpleUpSample, self).__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                out_channels + out_channels if not upsameple_only else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            ),
            get_act_layer(act_name),
            get_norm_layer(
                name=norm_name, spatial_dims=spatial_dims, channels=out_channels
            ),
        )

    def forward(self, inp, skip = None):
        out = self.transp_conv(inp)
        if skip is not None:
            out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SimpleCAUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        drop_path_rate=0.0,
        res_block: bool = True,
        add: bool = False,
    ) -> None:
        super(SimpleCAUnet, self).__init__()

        self.caformer = CAFormer(
            in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,
        )

        self.decoder1 = SimpleUpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder2 = SimpleUpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder3 = SimpleUpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        self.decoder4 = SimpleUpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0] // 4,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            upsameple_only=True
        )

        self.final_conv = get_conv_layer(
            spatial_dims,
            dims[0] // 4,
            out_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act,
        )

    def forward(self, x):
        x, hidden_states = self.caformer(x)

        y = hidden_states[2]

        x = self.decoder1(x, y)

        y = hidden_states[1]
        x = self.decoder2(x, y)  # /8

        y = hidden_states[0]  # /8
        x = self.decoder3(x, y)  # /4

        x = self.decoder4(x)

        x = self.final_conv(x)

        return x

class CAFormerPolyUnet(nn.Module):
    def __init__(
        self,
        in_channels,
        spatial_dims: int = 3,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        drop_path_rate=0.0,
        res_block: bool = True,
        add: bool = False,
    ) -> None:
        super(CAFormerPolyUnet, self).__init__()

        self.caformer = CAFormer(
            in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,
        )

        self.skip_encoder1 = MetaPolypConvFormerBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            norm_name=norm_name,
        )

        self.skip_encoder2 = MetaPolypConvFormerBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            norm_name=norm_name,
        )

        self.skip_encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.skip_encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder1 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder2 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder3 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder4 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0] // 4,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.final_conv = get_conv_layer(
            spatial_dims,
            dims[0] // 4,
            in_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act,
        )

    def forward(self, x):
        x, hidden_states = self.caformer(x)

        y = self.skip_encoder1(hidden_states[3])  # /32
        x = self.decoder1(x, y)  # /16

        y = self.skip_encoder2(hidden_states[2])  # /16
        x = self.decoder2(x, y)  # /8

        y = self.skip_encoder3(hidden_states[1])  # /8
        x = self.decoder3(x, y)  # /4

        y = self.skip_encoder4(hidden_states[0])  # /4
        x = self.decoder4(x, y)  # /1

        x = self.final_conv(x)

        return x


class CAFormerUnetWithoutSkip(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        drop_path_rate=0.0,
        res_block: bool = True,
        add: bool = False,
    ) -> None:
        super(CAFormerUnetWithoutSkip, self).__init__()

        self.caformer = CAFormer(
            in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,
        )

        self.decoder1 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder2 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder3 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.decoder4 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0] // 4,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=res_block,
            add=add,
        )

        self.final_conv = get_conv_layer(
            spatial_dims,
            dims[0] // 4,
            out_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act,
        )

    def forward(self, x):
        x, hidden_states = self.caformer(x)

        y = hidden_states[3] # /32
        x = self.decoder1(x, y)  # /16

        y = hidden_states[2] # /16
        x = self.decoder2(x, y)  # /8

        y = hidden_states[1]  # /8
        x = self.decoder3(x, y)  # /4

        y = hidden_states[0]  # /4
        x = self.decoder4(x, y)  # /1

        x = self.final_conv(x)

        return x

