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
        self.output_mlp = nn.Linear(dim * (num_clusters + 8), output_dim)
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

        # global geometry
        x_feat = self.x_conv(x)
        x_feat = x_feat.squeeze(-1).squeeze(-1)
        x_feat = F.normalize(x_feat, p=2, dim=1)  # L2 normalization

        # residuals weighting
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        vlad = residuals.sum(dim=-1)  # [N, num_clusters, C]

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        # output generation
        vlad = torch.cat([vlad, x_feat], dim=1)
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        return vlad


class YawNetVLAD(nn.Module):
    def __init__(self, dim=128, num_clusters=64, output_dim=256, centers_shared=True):
        super(YawNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.output_mlp = nn.Linear(dim * (num_clusters + 8), output_dim)
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

        # residuals weighting
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        residuals = residuals.view(N, self.num_clusters, self.dim, H, W)
        residuals = residuals.sum(dim=-2)

        residuals = torch.cat([residuals[..., 800:], residuals, residuals[..., :100]], dim=-1)

        residuals = residuals.unfold(dimension=-1, size=200, step=sample_step)

        residuals = residuals[:, :, :, :-1, :]

        vlad = residuals.sum(dim=-1)
        vlad = vlad.permute(0, 3, 1, 2)

        # fuse vlad and center-semantic
        b, n, _, _ = vlad.shape
        vlad = vlad.contiguous()
        vlad = vlad.view(b*n, self.num_clusters, self.dim)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        # x_feat = F.normalize(x_feat, p=2, dim=2)  # intra-normalization
        x_feat = x_feat.contiguous()
        x_feat = x_feat.view(b*n, -1)
        vlad = torch.cat([vlad, x_feat], dim=1)
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        vlad = vlad.view(b, n, -1)

        return vlad

