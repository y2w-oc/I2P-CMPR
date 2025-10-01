import torch
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
from .base_transformer import ResidualAttentionBlock
from .encoder import ResNetFCN, ResidualBlock


class VisualTransformer(nn.Module):
    def __init__(self, in_channel=3, embed_dim=128, num_att_head=8, patch_size=8, num_layer=5, dropout=0.1, input_size=(112, 600), fcn_ds_scale=[2, 2, 2, 3]):
        super().__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_att_head = num_att_head
        self.patch_size = patch_size
        self.encoder = ResNetFCN(in_channel=in_channel, dim=embed_dim, ds_scale=fcn_ds_scale)

        self.patch_partition = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)

        self.num_patches = (input_size[0] // (patch_size*fcn_ds_scale[3])) * (input_size[1] // (patch_size*fcn_ds_scale[3]))
        self.position_embeddings = nn.Parameter(data=self.get_sinusoid_embedding(n_position=self.num_patches, d_hid=embed_dim),
                                                requires_grad=False)

        self.res_att_blocks = nn.Sequential(
            *[ResidualAttentionBlock(embed_dim=embed_dim, num_head=num_att_head, dropout=dropout, att_mode='linear') for _ in range(num_layer)])

        self.upsample_trans_feat = torch.nn.Upsample(scale_factor=patch_size, mode='nearest')

        self.fuse_layer = ResidualBlock(2 * embed_dim, embed_dim, 5)
        self.out_layer = ResidualBlock(embed_dim, embed_dim, 5)

    def _get_position_angle_vec(self, position, d_hid):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    def get_sinusoid_embedding(self, n_position, d_hid):
        sinusoid_table = np.array([self._get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, img: torch.Tensor):
        cnn_feat = self.encoder(img)

        patches = self.patch_partition(cnn_feat)
        b, c, h, w = patches.shape
        patches = patches.view(b, c, -1)
        patches = patches.permute(0, 2, 1)

        patches = patches + self.position_embeddings
        patches = self.res_att_blocks(patches)

        patches = patches.permute(0, 2, 1)
        patches = patches.view(b, c, h, w)

        trans_feat = self.upsample_trans_feat(patches)

        fused_feat = torch.cat([cnn_feat, trans_feat], dim=1)
        fused_feat = self.fuse_layer(fused_feat)
        out = self.out_layer(fused_feat)

        return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class FullTransformer(nn.Module):
    def __init__(self, in_channel=3, patch_size=2, num_layer=5, dropout=0.1, input_size=(112, 600), ds_scale=[2, 2, 1, 3]):
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        topconv_scale = ds_scale[3]
        self.encoder = nn.Sequential(ResidualBlock(in_channel, 48, 5),
                                     ResidualBlock(48, 48, 5),
                                     nn.Conv2d(48, 64, kernel_size=topconv_scale, stride=topconv_scale, padding=0))

        dims = [96, 192, 384]

        self.patch_embed = nn.Conv2d(64, dims[0], kernel_size=patch_size, stride=patch_size, padding=0, bias=False)

        self.res_att_blocks_0 = nn.Sequential(*[ResidualAttentionBlock(embed_dim=dims[0], num_head=dims[0], dropout=dropout, att_mode='vanilla') for _ in range(4)])
        self.downsample0 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=ds_scale[0], stride=ds_scale[0], padding=0),
            LayerNorm2d(dims[1]),
        )

        self.res_att_blocks_1 = nn.Sequential(
            *[ResidualAttentionBlock(embed_dim=dims[1], num_head=dims[1], dropout=dropout, att_mode='vanilla') for _ in
              range(5)])
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=ds_scale[1], stride=ds_scale[1], padding=0),
            LayerNorm2d(dims[2]),
        )

        self.res_att_blocks_2 = nn.Sequential(
            *[ResidualAttentionBlock(embed_dim=dims[2], num_head=dims[2], dropout=dropout, att_mode='vanilla') for _ in
              range(4)])

        if patch_size == 2:
            K = 4
            S = 2
            P = 1
        elif patch_size == 3:
            K = 5
            S = 3
            P = 1
        else:
            raise ValueError("The patch_size should be 2 or 3!")

        self.patch_deconv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.ssm_up_layer1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.ssm_up_layer2 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[1], dims[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fuse_convs = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        )


    def get_sinusoid_embedding(self, n_position, d_hid):
        sinusoid_table = np.array([self._get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, img: torch.Tensor):
        # print("Input:", img.shape)
        cnn_feat = self.encoder(img)
        # print("Encoder:", cnn_feat.shape)

        patches = self.patch_embed(cnn_feat)
        # print("Patch:", patches.shape)

        vit_list = []

        # print("ViT0_Input:", patches.shape)
        b, c, h, w = patches.shape
        patches = patches.view(b, c, -1)
        patches = patches.permute(0, 2, 1)
        patches = self.res_att_blocks_0(patches)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(b, c, h, w)
        vit_list.append(patches)
        patches = self.downsample0(patches)
        # print("ViT0_Output:", patches.shape)

        # print("ViT1_Input:", patches.shape)
        b, c, h, w = patches.shape
        patches = patches.view(b, c, -1)
        patches = patches.permute(0, 2, 1)
        patches = self.res_att_blocks_1(patches)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(b, c, h, w)
        vit_list.append(patches)
        patches = self.downsample1(patches)
        # print("ViT1_Output:", patches.shape)

        # print("ViT2_Input:", patches.shape)
        b, c, h, w = patches.shape
        patches = patches.view(b, c, -1)
        patches = patches.permute(0, 2, 1)
        patches = self.res_att_blocks_2(patches)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(b, c, h, w)
        # print("ViT2_Output:", patches.shape)
        vit_list.append(patches)

        # print(vit_list[0].shape, vit_list[1].shape, vit_list[2].shape)

        vit_list[0] = self.patch_deconv(vit_list[0])
        vit_list[1] = self.ssm_up_layer1(vit_list[1])
        vit_list[2] = self.ssm_up_layer2(vit_list[2])
        vit_list.append(cnn_feat)

        fuse_res = torch.cat(vit_list, dim=1)
        out = self.fuse_convs(fuse_res)

        return out


class FullyConvNet(nn.Module):
    def __init__(self, in_channel=3, patch_size=2, num_layer=5, dropout=0.1, input_size=(112, 600), ds_scale=[2, 2, 1, 3]):
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        topconv_scale = ds_scale[3]
        self.encoder = nn.Sequential(ResidualBlock(in_channel, 48, 5),
                                     ResidualBlock(48, 48, 5),
                                     nn.Conv2d(48, 64, kernel_size=topconv_scale, stride=topconv_scale, padding=0))

        dims = [96, 192, 384]

        self.downsample = nn.Conv2d(64, dims[0], kernel_size=patch_size, stride=patch_size, padding=0, bias=False)

        self.res_att_blocks_0 = nn.Sequential(*[ResidualBlock(dims[0], dims[0], 5) for _ in range(2)])
        self.downsample0 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=ds_scale[0], stride=ds_scale[0], padding=0),
            LayerNorm2d(dims[1]),
        )

        self.res_att_blocks_1 = nn.Sequential(*[ResidualBlock(dims[1], dims[1], 5) for _ in range(3)])
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=ds_scale[1], stride=ds_scale[1], padding=0),
            LayerNorm2d(dims[2]),
        )

        self.res_att_blocks_2 = nn.Sequential(*[ResidualBlock(dims[2], dims[2], 5) for _ in range(2)])

        if patch_size == 2:
            K = 4
            S = 2
            P = 1
        elif patch_size == 3:
            K = 5
            S = 3
            P = 1
        else:
            raise ValueError("The patch_size should be 2 or 3!")

        self.patch_deconv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.ssm_up_layer1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.ssm_up_layer2 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[1], dims[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(dims[0], 64, kernel_size=K, stride=S, padding=P, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fuse_convs = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        )


    def get_sinusoid_embedding(self, n_position, d_hid):
        sinusoid_table = np.array([self._get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, img: torch.Tensor):
        # print("Input:", img.shape)
        cnn_feat = self.encoder(img)
        # print("Encoder:", cnn_feat.shape)

        patches = self.downsample(cnn_feat)
        # print("Patch:", patches.shape)

        vit_list = []

        # print("ViT0_Input:", patches.shape)
        patches = self.res_att_blocks_0(patches)
        vit_list.append(patches)
        patches = self.downsample0(patches)
        # print("ViT0_Output:", patches.shape)

        # print("ViT1_Input:", patches.shape)
        patches = self.res_att_blocks_1(patches)
        vit_list.append(patches)
        patches = self.downsample1(patches)
        # print("ViT1_Output:", patches.shape)

        # print("ViT2_Input:", patches.shape)
        patches = self.res_att_blocks_2(patches)
        # print("ViT2_Output:", patches.shape)
        vit_list.append(patches)

        # print(vit_list[0].shape, vit_list[1].shape, vit_list[2].shape)

        vit_list[0] = self.patch_deconv(vit_list[0])
        vit_list[1] = self.ssm_up_layer1(vit_list[1])
        vit_list[2] = self.ssm_up_layer2(vit_list[2])
        vit_list.append(cnn_feat)

        fuse_res = torch.cat(vit_list, dim=1)
        out = self.fuse_convs(fuse_res)

        return out


if __name__ == '__main__':
    device = 'cuda:0'

    before_memory = torch.cuda.memory_allocated()
    model1 = FullTransformer(in_channel=3, patch_size=2, num_layer=5, dropout=0.1, input_size=(120, 600), ds_scale=[2, 2, 1, 3])
    model1 = model1.to(device)

    model2 = FullTransformer(in_channel=1, patch_size=3, num_layer=5, dropout=0.1, input_size=(120, 600), ds_scale=[2, 2, 1, 1])
    model2 = model2.to(device)

    # input1 = torch.randn((1, 3, 120, 600), device=next(model1.parameters()).device)
    # out1 = model1(input1)
    # print(input1.shape, out1.shape)

    input2 = torch.randn((1, 1, 48, 900), device=next(model2.parameters()).device)
    out2 = model2(input2)

    print(input2.shape, out2.shape)
