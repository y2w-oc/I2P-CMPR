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