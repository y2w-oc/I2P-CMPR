import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, dim=128, num_clusters=64, output_dim=256, centers_shared=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.output_mlp = nn.Linear(dim * 8, output_dim)
        with torch.no_grad():
            self.conv.weight.div_(math.sqrt(dim))
            self.conv.bias.div_(math.sqrt(dim))
            self.output_mlp.weight.div_(math.sqrt(dim))
            self.output_mlp.bias.div_(math.sqrt(dim))

        self.centers_shared = centers_shared
        if centers_shared:
            pass
        else:
            self.centroids = nn.Parameter(torch.rand(num_clusters, dim) / math.sqrt(dim))

        # positional embeddings
        self.x_conv = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size=5, stride=1, padding=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
                                    nn.Conv2d(dim*2, dim*4, kernel_size=5, stride=1, padding=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                    nn.Conv2d(dim*4, dim*8, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=(5, 25))
                                    )

    def forward(self, x, cluster_centers=None):
        if self.centers_shared:
            assert cluster_centers is not None, "Missing the shared clustering centers (input)!"

        # global geometry
        x_feat = self.x_conv(x)
        x_feat = x_feat.squeeze(-1).squeeze(-1)
        x_feat = F.normalize(x_feat, p=2, dim=1)  # L2 normalization


        # output generation
        vlad = x_feat
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        return vlad


class YawNetVLAD(nn.Module):
    def __init__(self, dim=128, num_clusters=64, output_dim=256, centers_shared=True):
        super(YawNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.output_mlp = nn.Linear(dim * 8, output_dim)
        with torch.no_grad():
            self.conv.weight.div_(math.sqrt(dim))
            self.conv.bias.div_(math.sqrt(dim))
            self.output_mlp.weight.div_(math.sqrt(dim))
            self.output_mlp.bias.div_(math.sqrt(dim))

        self.centers_shared = centers_shared
        if centers_shared:
            pass
        else:
            self.centroids = nn.Parameter(torch.rand(num_clusters, dim) / math.sqrt(dim))

        # positional embeddings
        self.x_conv = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size=5, stride=1, padding=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 1), padding=(0, 1)),
                                    nn.Conv2d(dim*2, dim*4, kernel_size=5, stride=1, padding=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.MaxPool2d(kernel_size=(4, 5), stride=(4, 1), padding=(0, 2)),
                                    nn.Conv2d(dim*4, dim*8, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=(4, 200), stride=(4, 1), padding=(0, 0)),
                                    )

    def forward(self, x, cluster_centers=None):
        if self.centers_shared:
            assert cluster_centers is not None, "Missing the shared clustering centers (input)!"

        if self.training:
            sample_step = 10
        else:
            sample_step = 30

        # global geometry
        x = torch.cat([x[..., 800:], x, x[..., :100]], dim=-1)
        x_feat = self.x_conv(x)
        x_feat = x_feat[:, :, 0, ::sample_step]
        x_feat = x_feat[..., :-1]
        x_feat = x_feat.permute(0, 2, 1)
        x_feat = F.normalize(x_feat, p=2, dim=2)  # L2 normalization

        # x_feat = F.normalize(x_feat, p=2, dim=2)  # intra-normalization
        x_feat = x_feat.contiguous()
        b,n,_ = x_feat.shape
        x_feat = x_feat.view(b*n, -1)
        vlad = x_feat
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        vlad = vlad.view(b, n, -1)

        return vlad

