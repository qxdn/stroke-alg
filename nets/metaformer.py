import torch.nn as nn
import torch
from typing import Tuple, Union, Sequence
from layers import Stem, MetaFormerStage
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
    ) -> None:
        super(UpSample, self).__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels + in_channels,
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
                res_block=res_block
            )
       
    def forward(self,inp, skip):
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
        in_channels,
        spatial_dims: int = 3,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        norm_name: str = "instance",
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        drop_path_rate=0.0,
        res_block: bool = True,
    ) -> None:
        super(CAFormerUnet, self).__init__()

        self.caformer = CAFormer(
            in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,
        )

        skip_encoders = [
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            for dim in dims
        ]
        skip_encoders.reverse()

        self.skip_encoders = nn.ModuleList(skip_encoders)

        self.decoders = nn.ModuleList(
            [
                UpSample(
                    spatial_dims=spatial_dims,
                    in_channels=dims[-i],
                    out_channels=dims[-i - 1],
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=res_block,
                )
                for i in range(1, len(dims))
            ]
        )

        self.last_upsample = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0],
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.final_conv = get_conv_layer(
            spatial_dims,
            dims[0],
            in_channels,
            kernel_size=1,
            stride=1,
            norm=norm_name,
            act=act,

        )

    def forward(self, x):
        x, hidden_states = self.caformer(x)
        hidden_states.reverse()

        for decoder, hidden_state, skip_encoder in zip(
            self.decoders, hidden_states, self.skip_encoders
        ):
            skip = skip_encoder(hidden_state)
            x = decoder(x , skip)
        
        x = self.last_upsample(x)
        x = self.final_conv(x)

        return x
