import torch
import torch.nn.functional as F
from torch import nn
import math
import time
import numpy as np
from .visual_transformer import VisualTransformer, FullTransformer, FullyConvNet
import torch.nn.init as init
import matplotlib.pyplot as plt
from .netvlad import NetVLAD, YawNetVLAD
import os
from models.mamba import build_mamba_model


class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 64
        self.num_clusters = 48
        self.netvlad_dim = 256
        self.num_words = 1024
        self.shared_centers = True

        self.rgb_model = FullTransformer(in_channel=3, patch_size=2, num_layer=5, dropout=0.1, input_size=(120, 600), ds_scale=[2, 2, 1, 3])


        self.range_model = FullTransformer(in_channel=1, patch_size=3, num_layer=5, dropout=0.1, input_size=(120, 600), ds_scale=[2, 2, 1, 1])


        self.shared_cluster_centers = nn.Parameter(torch.rand(self.num_clusters, self.dim) / math.sqrt(self.dim))
        self.rgb_net_vlad = NetVLAD(dim=self.dim, num_clusters=self.num_clusters, output_dim=self.netvlad_dim,
                                    centers_shared=self.shared_centers)
        self.range_net_vlad = YawNetVLAD(dim=self.dim, num_clusters=self.num_clusters, output_dim=self.netvlad_dim,
                                         centers_shared=self.shared_centers)

    # @torch.no_grad()
    def rgb_desc_inference(self, rgb_imgs):
        # torch.save(self.rgb_model.state_dict(), "rgb_model.pth")
        rgb_feat = self.rgb_model(rgb_imgs)
        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        if self.shared_centers:
            rgb_descriptor = self.rgb_net_vlad(rgb_feat, self.shared_cluster_centers)
        else:
            rgb_descriptor = self.rgb_net_vlad(rgb_feat)
        # ddd = rgb_descriptor.cpu().data.numpy()
        # print(ddd.shape)
        # plt.plot(ddd[0])
        # plt.show()
        # torch.save(self.rgb_net_vlad.state_dict(), "rgb_net_vlad.pth")
        return rgb_descriptor

    # @torch.no_grad()
    def range_desc_inference(self, range_imgs):
        # torch.save(self.range_model.state_dict(), "range_model.pth")
        range_feat = self.range_model(range_imgs)
        range_feat = F.normalize(range_feat, p=2, dim=1)
        if self.shared_centers:
            range_descriptor = self.range_net_vlad(range_feat, self.shared_cluster_centers)
        else:
            range_descriptor = self.range_net_vlad(range_feat)
        # torch.save(self.range_net_vlad.state_dict(), "range_net_vlad.pth")
        return range_descriptor

    def set_mamba_static(self):
        self.rgb_model.eval()
        self.range_model.eval()

    def forward(self, rgb_imgs=None, range_imgs=None):
        # with torch.no_grad():
        rgb_feat = self.rgb_model(rgb_imgs)
        range_feat = self.range_model(range_imgs)

        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        range_feat = F.normalize(range_feat, p=2, dim=1)

        if self.shared_centers:
            rgb_descriptor = self.rgb_net_vlad(rgb_feat, self.shared_cluster_centers)
            range_descriptor = self.range_net_vlad(range_feat, self.shared_cluster_centers)
        else:
            rgb_descriptor = self.rgb_net_vlad(rgb_feat)
            range_descriptor = self.range_net_vlad(range_feat)

        return rgb_feat, range_feat, rgb_descriptor, range_descriptor
