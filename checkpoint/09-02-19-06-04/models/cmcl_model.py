import torch
import torch.nn.functional as F
from torch import nn
import math
import time
import numpy as np
from .visual_transformer import VisualTransformer
import torch.nn.init as init
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

        self.rgb_model = build_mamba_model(mamba_version="./models/mamba/vmambav2_tiny_224.yaml", mamba_patch=2, ds_scale=[2,2,1], topconv_scale=3, inchans=3)

        # self.rgb_model = VisualTransformer(in_channel=3,
        #                                    embed_dim=self.dim,
        #                                    num_att_head=8,
        #                                    patch_size=5,
        #                                    num_layer=5,
        #                                    dropout=0.1,
        #                                    input_size=(120, 600),
        #                                    fcn_ds_scale=[2, 2, 2, 3])

        # load pretrained model
        # checkpoint_path = os.getcwd()+"/rgb_model.pth"
        # print("Checkpoint " + checkpoint_path + " Loading!")
        # checkpoint = torch.load(checkpoint_path)
        # self.rgb_model.load_state_dict(checkpoint)

        self.range_model = build_mamba_model(mamba_version="./models/mamba/vmambav2_tiny_224.yaml", mamba_patch=3, ds_scale=[2,2,1], topconv_scale=1, inchans=1)

        # self.range_model = VisualTransformer(in_channel=1,
        #                                      embed_dim=self.dim,
        #                                      num_att_head=8,
        #                                      patch_size=6,
        #                                      num_layer=5,
        #                                      dropout=0.1,
        #                                      input_size=(48, 900),
        #                                      fcn_ds_scale=[2, 2, 3, 1])

        # load pretrained model
        # checkpoint_path = os.getcwd()+"/range_model.pth"
        # print("Checkpoint " + checkpoint_path + " Loading!")
        # checkpoint = torch.load(checkpoint_path)
        # self.range_model.load_state_dict(checkpoint)

        self.shared_cluster_centers = nn.Parameter(torch.rand(self.num_clusters, self.dim) / math.sqrt(self.dim))
        self.rgb_net_vlad = NetVLAD(dim=self.dim, num_clusters=self.num_clusters, output_dim=self.netvlad_dim,
                                    centers_shared=True)
        self.range_net_vlad = YawNetVLAD(dim=self.dim, num_clusters=self.num_clusters, output_dim=self.netvlad_dim,
                                         centers_shared=True)

    # @torch.no_grad()
    def rgb_desc_inference(self, rgb_imgs):
        # torch.save(self.rgb_model.state_dict(), "rgb_model.pth")
        rgb_feat = self.rgb_model(rgb_imgs)
        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        rgb_descriptor = self.rgb_net_vlad(rgb_feat, self.shared_cluster_centers)
        # torch.save(self.rgb_net_vlad.state_dict(), "rgb_net_vlad.pth")
        return rgb_descriptor

    # @torch.no_grad()
    def range_desc_inference(self, range_imgs):
        # torch.save(self.range_model.state_dict(), "range_model.pth")
        range_feat = self.range_model(range_imgs)
        range_feat = F.normalize(range_feat, p=2, dim=1)
        range_descriptor = self.range_net_vlad(range_feat, self.shared_cluster_centers)
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

        rgb_descriptor = self.rgb_net_vlad(rgb_feat, self.shared_cluster_centers)
        range_descriptor = self.range_net_vlad(range_feat, self.shared_cluster_centers)

        return rgb_feat, range_feat, rgb_descriptor, range_descriptor
