#@title Generator
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_channels, 1)
        nn.BatchNorm2d(n_channels)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = torch.tanh(self.conv_last(x))
        # out = torch.sigmoid(self.conv_last(x))

        return out




# class ShadingUNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()

#         self.dconv_down1 = double_conv(3, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(256 + 512, 256)
#         self.dconv_up2 = double_conv(128 + 256, 128)
#         self.dconv_up1 = double_conv(128 + 64, 64)

#         self.conv_last = nn.Conv2d(64, n_channels, 1)


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x = self.dconv_up3(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv2], dim=1)

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv1], dim=1)

#         x = self.dconv_up1(x)

#         # out = torch.tanh(self.conv_last(x))
#         out = 0.5 + torch.sigmoid(self.conv_last(x))
#         # out = 1. - torch.tanh(self.conv_last(x))
#         # out = self.conv_last(x)
#         return out.repeat([1, 3, 1, 1])


class ShadingUNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(64 + 128, 64)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)

        self.conv_last = nn.Conv2d(16, 1, 1)
        self.n_channels = n_channels


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = 2 * torch.sigmoid(self.conv_last(x))
        return out.repeat([1, self.n_channels, 1, 1])


class AlphaUNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.dconv_down1 = double_conv(n_channels, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(64 + 128, 64)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)

        self.conv_last = nn.Conv2d(16, n_channels + 1, 1)
        self.n_channels = n_channels


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        out_layer = torch.sigmoid(out[:, :self.n_channels])
        alpha_layer = torch.sigmoid(out[:, -1])
        return out_layer, alpha_layer





class TwoLayerUNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3_mult = double_conv(64 + 128, 64)
        self.dconv_up2_mult = double_conv(32 + 64, 32)
        self.dconv_up1_mult = double_conv(32 + 16, 16)

        self.dconv_up3_div = double_conv(64 + 128, 64)
        self.dconv_up2_div = double_conv(32 + 64, 32)
        self.dconv_up1_div = double_conv(32 + 16, 16)

        self.conv_last_mult = nn.Conv2d(16, 1, 1)
        self.conv_last_div = nn.Conv2d(16, 1, 1)
        self.n_channels = n_channels


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x_mult = self.dconv_up3_mult(x)
        x_mult = self.upsample(x_mult)
        x_mult = torch.cat([x_mult, conv2], dim=1)

        x_div = self.dconv_up3_div(x)
        x_div = self.upsample(x_div)
        x_div = torch.cat([x_div, conv2], dim=1)

        x_mult = self.dconv_up2_mult(x_mult)
        x_mult = self.upsample(x_mult)
        x_mult = torch.cat([x_mult, conv1], dim=1)

        x_div = self.dconv_up2_div(x_div)
        x_div = self.upsample(x_div)
        x_div = torch.cat([x_div, conv1], dim=1)

        x_mult = self.dconv_up1_mult(x_mult)
        x_div = self.dconv_up1_div(x_div)

        # out = torch.sigmoid(self.conv_last(x))
        out_mult = torch.sigmoid(self.conv_last_mult(x_mult))
        out_div = torch.sigmoid(self.conv_last_div(x_div))
        out_div = torch.clip(out_div, 0.1, 1.)
        out_mult = torch.clip(out_mult, 0.1, 1.)
        return out_mult.repeat([1, self.n_channels, 1, 1]), out_div.repeat([1, self.n_channels, 1, 1])