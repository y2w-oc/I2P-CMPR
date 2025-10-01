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
        self.output_mlp = nn.Linear(dim * num_clusters * 2, output_dim)
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
        self.map_conv = nn.Sequential(nn.Conv2d(1, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      torch.nn.MaxPool2d((40, 200))
                                      )

    def forward(self, x, cluster_centers=None):
        if self.centers_shared:
            assert cluster_centers is not None, "Missing the shared clustering centers (input)!"

        N, C, H, W = x.shape  # N: batch size, C: number of channels (dim), H: height, W: width

        # soft-assignment
        soft_assign = self.conv(x)  # [N, num_clusters, H, W]
        soft_assign = soft_assign.view(N, self.num_clusters, -1)  # [N, num_clusters, H*W]
        soft_assign = F.softmax(soft_assign, dim=1)  # [N, num_clusters, H*W]

        x_flatten = x.view(N, C, -1)  # [N, C, H*W]

        if self.centers_shared:
            residuals = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                        cluster_centers.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)  # [N, num_clusters, C, H*W]
        else:
            residuals = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                        self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)  # [N, num_clusters, C, H*W]

        # positional embeddings
        residuals_temp = residuals.view(N*self.num_clusters, C, H, W)
        pixel2center_map = torch.norm(residuals_temp, p=2, dim=1, keepdim=True)
        pixel2center_map = pixel2center_map.detach()
        map_feat = self.map_conv(pixel2center_map)
        map_feat = map_feat.view(N, self.num_clusters, C)

        # residuals weighting
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        vlad = residuals.sum(dim=-1)  # [N, num_clusters, C]

        # fuse vlad and center-semantic
        map_feat = F.normalize(map_feat, p=2, dim=2)  # intra-normalization
        map_feat = map_feat.view(map_feat.size(0), -1)  # flatten
        map_feat = F.normalize(map_feat, p=2, dim=1)  # L2 normalization

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        # output generation
        vlad = torch.cat([vlad, map_feat], dim=1)
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        return vlad


class YawNetVLAD(nn.Module):
    def __init__(self, dim=128, num_clusters=64, output_dim=256, centers_shared=True):
        super(YawNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.output_mlp = nn.Linear(dim * num_clusters * 2, output_dim)
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
        self.map_conv = nn.Sequential(nn.Conv2d(1, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
                                      )

        self.map_maxpool = torch.nn.MaxPool2d(kernel_size=(48, 200), stride=(1, 1), padding=0)

    def forward(self, x, cluster_centers=None):
        if self.centers_shared:
            assert cluster_centers is not None, "Missing the shared clustering centers (input)!"

        N, C, H, W = x.shape  # N: batch size, C: number of channels (dim), H: height, W: width

        # soft-assignment
        soft_assign = self.conv(x)  # [N, num_clusters, H, W]
        soft_assign = soft_assign.view(N, self.num_clusters, -1)  # [N, num_clusters, H*W]
        soft_assign = F.softmax(soft_assign, dim=1)  # [N, num_clusters, H*W]

        x_flatten = x.view(N, C, -1)  # [N, C, H*W]

        if self.centers_shared:
            residuals = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                        cluster_centers.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)  # [N, num_clusters, C, H*W]
        else:
            residuals = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                        self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)  # [N, num_clusters, C, H*W]

        # positional embeddings
        residuals_temp = residuals.view(N * self.num_clusters, C, H, W)
        pixel2center_map = torch.norm(residuals_temp, p=2, dim=1, keepdim=True)
        pixel2center_map = pixel2center_map.detach()
        map_feat = self.map_conv(pixel2center_map)
        map_feat = torch.cat([map_feat[..., 800:], map_feat, map_feat[..., :100]], dim=-1)
        map_feat = self.map_maxpool(map_feat).squeeze(-2)
        map_feat = map_feat.view(N, self.num_clusters, C, 901)
        map_feat = map_feat[:, :, :, ::10]
        map_feat = map_feat[..., :-1]

        # residuals weighting
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        residuals = residuals.view(N, self.num_clusters, self.dim, H, W)
        residuals = residuals.sum(dim=-2)

        residuals = torch.cat([residuals[..., 800:], residuals, residuals[..., :100]], dim=-1)

        if self.training:
            residuals = residuals.unfold(dimension=-1, size=200, step=10)
        else:
            residuals = residuals.unfold(dimension=-1, size=200, step=10)

        residuals = residuals[:, :, :, :-1, :]

        vlad = residuals.sum(dim=-1)
        vlad = vlad.permute(0, 3, 1, 2)

        # fuse vlad and center-semantic
        b, n, _, _ = vlad.shape
        vlad = vlad.contiguous()
        vlad = vlad.view(b*n, self.num_clusters, self.dim)

        map_feat = map_feat.permute(0, 3, 1, 2)
        map_feat = map_feat.contiguous()
        map_feat = map_feat.view(b*n, self.num_clusters, self.dim)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        map_feat = F.normalize(map_feat, p=2, dim=2)  # intra-normalization
        map_feat = map_feat.view(map_feat.size(0), -1)  # flatten
        map_feat = F.normalize(map_feat, p=2, dim=1)  # L2 normalization

        vlad = torch.cat([vlad, map_feat], dim=1)
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        vlad = vlad.view(b, n, -1)

        return vlad

