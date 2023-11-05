from typing import Tuple, Union, List
import torch.nn as nn
import torch
from layers import DSConv3dBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode, InterpolateMode
import einops


class DSSegResNet(nn.Module):
    """
    SegResNet with DSConv3dBlock
    """

    def __init__(
        self,
        lesion_in_channels: int,
        blood_in_channels: int,
        out_channels: int,
        init_filters: int = 8,
        dropout_rate: float = 0.0,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        use_conv_final: bool = True,
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ) -> None:
        super(DSSegResNet, self).__init__()

        self.init_filters = init_filters
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.spatial_dims = 3

        self.lesion_conv_init = Convolution(
            self.spatial_dims,
            lesion_in_channels,
            init_filters,
            bias=False,
            conv_only=True,
        )
        self.blood_conv_init = Convolution(
            self.spatial_dims,
            blood_in_channels,
            init_filters,
            bias=False,
            conv_only=True,
        )
        self.lesion_down_layers = self._make_lesion_down_layers()
        self.blood_down_layers = self._make_blood_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

    def _make_lesion_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, filters, spatial_dims, dropout_rate = (
            self.blocks_down,
            self.init_filters,
            self.spatial_dims,
            self.dropout_rate,
        )
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                Convolution(
                    spatial_dims,
                    layer_in_channels // 2,
                    layer_in_channels,
                    kernel_size=3,
                    strides=2,
                    bias=False,
                    conv_only=True,
                )
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv,
                *[
                    DSConv3dBlock(
                        layer_in_channels,
                        layer_in_channels,
                        res_block=True,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(item)
                ],
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_blood_down_layers(self):
        return self._make_lesion_down_layers()

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        DSConv3dBlock(sample_in_channels // 2, sample_in_channels // 2)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        Convolution(
                            spatial_dims,
                            sample_in_channels,
                            sample_in_channels // 2,
                            kernel_size=1,
                            bias=False,
                            conv_only=True,
                        ),
                        UpSample(
                            spatial_dims=spatial_dims,
                            in_channels=sample_in_channels // 2,
                            out_channels=sample_in_channels // 2,
                            scale_factor=2,
                            mode=upsample_mode,
                            interp_mode=InterpolateMode.LINEAR,
                            align_corners=False,
                        ),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels):
        return Convolution(
            self.spatial_dims,
            self.init_filters,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def encode_lesion(
        self, lesion: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.lesion_conv_init(lesion)

        down_x = []
        for down in self.lesion_down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def encode_blood(
        self, blood: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.blood_conv_init(blood)

        down_x = []
        for down in self.blood_down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(
        self,
        x: torch.Tensor,
        down_lesion: List[torch.Tensor],
        down_blood: List[torch.Tensor],
    ) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_lesion[i + 1] + down_blood[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, lesion, blood):
        lesion, down_lesion = self.encode_lesion(lesion)
        down_lesion.reverse()
        blood, down_blood = self.encode_blood(blood)
        down_blood.reverse()
        x = lesion + blood
        x = self.decode(x, down_lesion, down_blood)

        return x


class DSSegResNetWrapper(nn.Module):
    def __init__(
        self,
        lesion_in_channels: int,
        blood_in_channels: int,
        out_channels: int,
        init_filters: int = 8,
        dropout_rate: float = 0.0,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        use_conv_final: bool = True,
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ) -> None:
        super(DSSegResNetWrapper, self).__init__()

        self.lesion_in_channels = lesion_in_channels
        self.blood_in_channels = blood_in_channels

        self.model = DSSegResNet(
            lesion_in_channels=lesion_in_channels,
            blood_in_channels=blood_in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            dropout_rate=dropout_rate,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            use_conv_final=use_conv_final,
            upsample_mode=upsample_mode,
        )

    def forward(self, x):
        lesion = x[:, : self.lesion_in_channels]
        blood = x[:, self.lesion_in_channels :]
        return self.model(lesion, blood)
