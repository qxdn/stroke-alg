import torch.nn as nn
import torch
from layers import DSConv3d


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        extend_scope: int = 1,
        if_offset: bool = True,
        encoder: bool = True,
    ) -> None:
        super(DSConv, self).__init__()

        self.conv = (
            EncoderConv(in_ch, out_ch) if encoder else DecoderConv(in_ch, out_ch)
        )
        self.ds_conv_x = DSConv3d(
            in_ch, out_ch, kernel_size, extend_scope, 0, if_offset
        )
        self.ds_conv_y = DSConv3d(
            in_ch, out_ch, kernel_size, extend_scope, 1, if_offset
        )
        self.ds_conv_z = DSConv3d(
            in_ch, out_ch, kernel_size, extend_scope, 2, if_offset
        )

        self.out_conv = (
            EncoderConv(4*out_ch, out_ch) if encoder else DecoderConv(4*out_ch, out_ch)
        )

    def forward(self, inputs):
        conv_out = self.conv(inputs)
        x = self.ds_conv_x(inputs)
        y = self.ds_conv_y(inputs)
        z = self.ds_conv_z(inputs)

        out = torch.cat([conv_out, x, y, z], dim=1)
        out = self.out_conv(out)

        return out


class DSCNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        filters: int = 16,
        kernel_size: int = 3,
        extend_scope: int = 1,
        if_offset: bool = True,
    ):
        super(DSCNet, self).__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.filters = filters

        # Unet
        self.conv1 = DSConv(
            n_channels,
            self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
        )

        self.conv2 = DSConv(
            self.filters,
            2 * self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
        )

        self.conv3 = DSConv(
            2 * self.filters,
            4 * self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
        )

        self.conv4 = DSConv(
            4 * self.filters,
            8 * self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
        )

        self.conv5 = DSConv(
            12 * self.filters,
            4 * self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
        )

        self.conv6 = DSConv(
            6 * self.filters,
            2 * self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
            encoder=False,
        )

        self.conv7 = DSConv(
            3 * self.filters,
            self.filters,
            self.kernel_size,
            self.extend_scope,
            self.if_offset,
            encoder=False,
        )

        self.out_conv = nn.Conv3d(self.filters, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        # x = self.maxpooling(x)
        x1 = self.conv1(x)

        # block1
        x = self.maxpooling(x1)
        x2 = self.conv2(x)

        # block2
        x = self.maxpooling(x2)
        x3 = self.conv3(x)

        # block3
        x = self.maxpooling(x3)
        x4 = self.conv4(x)

        # block4
        x = self.up(x4)
        x5 = self.conv5(torch.cat([x, x3], dim=1))

        # block5
        x = self.up(x5)
        x6 = self.conv6(torch.cat([x, x2], dim=1))

        # block6
        x = self.up(x6)
        x7 = self.conv7(torch.cat([x, x1], dim=1))

        out = self.out_conv(x7)
        # out = self.softmax(out)
        # out = self.up(out)

        return out
