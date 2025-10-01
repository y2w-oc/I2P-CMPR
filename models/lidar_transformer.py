#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for KITTI sequences


import os
import sys
import time
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn

from .netvlad_gate import NetVLADLoupe
import torch.nn.functional as F
import time
import yaml

"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
"""


class LiDARTransformer(nn.Module):
    def __init__(self, input_channels=5):
        super(LiDARTransformer, self).__init__()

        """
            Learning on Longitudinal Plane
        """
        self.Longitudinal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(5,1), stride=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(2, 1), stride=(2, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False),
            nn.ReLU(inplace=True),
            )

        self.L_bridge_encoder_transformer = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False),
                                                          nn.ReLU(inplace=True))

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True,dropout=0.)
        self.Longitudinal_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.L_bridge_transformer_NetVlad = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False),
                                                          nn.ReLU(inplace=True)
                                                          )

        self.L_net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

    def forward(self, x):
        """
            Learning on Longitudinal Plane
        """
        out_l = self.Longitudinal_encoder(x)

        out_l_1 = out_l.permute(0,1,3,2)
        out_l_1 = self.L_bridge_encoder_transformer(out_l_1)

        out_l = out_l_1.squeeze(3)
        out_l = out_l.permute(0, 2, 1)

        out_l = self.Longitudinal_transformer(out_l)

        out_l = out_l.permute(0, 2, 1)

        out_l = out_l.unsqueeze(3)
        out_l = torch.cat((out_l_1, out_l), dim=1)

        out_l = self.L_bridge_transformer_NetVlad(out_l)

        out_l = F.normalize(out_l, dim=1)
        out_l = self.L_net_vlad(out_l)
        out_l = F.normalize(out_l, dim=1)

        return out_l

