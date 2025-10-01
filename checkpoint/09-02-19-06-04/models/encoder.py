import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel=3, outchannel=64, kernel=3):
        super(ResidualBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel, stride=1, padding=int(kernel/2)),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel, stride=1, padding=int(kernel/2)),
            nn.BatchNorm2d(outchannel)
        )
        self.final_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_layers(x) + self.shortcut(x)
        return self.final_relu(out)


class ResNetFCN(nn.Module):
    def __init__(self, in_channel=3, dim=128, ds_scale=[2, 2, 2, 3]):
        super(ResNetFCN, self).__init__()
        self.ds_scale = ds_scale
        # Encoder
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_layer1 = ResidualBlock(dim, dim, 5)
        self.conv_layer2 = ResidualBlock(dim, dim, 5)
        self.conv_layer3 = ResidualBlock(dim, dim, 5)
        self.conv_layer4 = ResidualBlock(dim, dim, 5)

        self.maxpoolx2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpoolx3 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=0)

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        if ds_scale[2] == 2:
            self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # nn.UpsamplingNearest2d(scale_factor=4),
            )
        elif ds_scale[2] == 3:
            self.deconv_layer4 = nn.Sequential(
                nn.Upsample(scale_factor=(2, 3), mode='nearest'),
                nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            raise ValueError("The down-sampling scales (ds_sacle: List) should match with the input images!")

        if ds_scale[3] == 1:
            self.ms_fuse_layer = ResidualBlock(4*dim, dim, 5)
        elif ds_scale[3] == 3:
            self.ms_fuse_layer = nn.Sequential(*[ResidualBlock(4 * dim, dim, 5),
                                                 nn.Conv2d(dim, dim, kernel_size=3, stride=3, padding=0)])
        else:
            raise ValueError("The output down-sampling scale (ds_sacle[3]: int) should be 1 or 3!")

    def forward(self, img):
        # Encoder
        conv1_out = self.pre_conv(img)
        layer1_out = self.conv_layer1(conv1_out)
        # print("layer1_out:", layer1_out.shape)
        layer2_out = self.conv_layer2(self.maxpoolx2(layer1_out))
        # print("layer2_out:", layer2_out.shape)
        layer3_out = self.conv_layer3(self.maxpoolx2(layer2_out))
        # print("layer3_out:", layer3_out.shape)
        if self.ds_scale[2] == 2:
            layer4_out = self.conv_layer4(self.maxpoolx2(layer3_out))
        elif self.ds_scale[2] == 3:
            layer4_out = self.conv_layer4(self.maxpoolx3(layer3_out))
        else:
            raise ValueError("The downsampling scales (ds_sacle: List) should match with the input images!")
        # print("layer4_out:", layer4_out.shape)
        # Decoder
        layer4_out = self.deconv_layer4(layer4_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer1_out = self.deconv_layer1(layer1_out)

        ms_out = torch.cat([layer1_out, layer2_out, layer3_out, layer4_out], dim=1)

        out = self.ms_fuse_layer(ms_out)

        return out
