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
        self.output_mlp = nn.Linear(dim * num_clusters, output_dim)
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
        img_shape = (40, 200)
        self.pe = nn.Parameter(data=self.positional_embedding(dim, img_shape), requires_grad=False)

        self.pos_fuse = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2))

    @staticmethod
    def positional_embedding(dim=64, img_shape=(100, 100)):
        pe = torch.zeros((dim, *img_shape))
        y_position = torch.ones(img_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(img_shape).cumsum(1).float().unsqueeze(0)

        div_term = torch.exp(
            torch.arange(0, dim // 2, 2).float() * (-math.log(img_shape[0] * img_shape[1]) / (dim // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        pe = pe.view(dim, -1)
        return pe.unsqueeze(0)

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
        residuals = residuals.view(N*self.num_clusters, C, H, W)
        print(residuals.shape, self.pe.shape)
        pos = self.pe.view(1, C, H, W)
        res_pos = residuals + pos
        residuals = self.pos_fuse(res_pos)
        residuals = residuals.view(N, self.num_clusters, C, -1)
        print(residuals.shape, self.pe.shape, soft_assign.shape)

        # weighting
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        vlad = residuals.sum(dim=-1)  # [N, num_clusters, C]

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        # output generation
        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        return vlad


class YawNetVLAD(nn.Module):
    def __init__(self, dim=128, num_clusters=64, output_dim=256, centers_shared=True):
        super(YawNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.output_mlp = nn.Linear(dim * num_clusters, output_dim)
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
        img_shape = (48, 900)
        self.pe = nn.Parameter(data=self.positional_embedding(dim, img_shape), requires_grad=False)

        self.pos_fuse = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(dim),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2))

    @staticmethod
    def positional_embedding(dim=64, img_shape=(100, 100)):
        pe = torch.zeros((dim, *img_shape))
        y_position = torch.ones(img_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(img_shape).cumsum(1).float().unsqueeze(0)

        div_term = torch.exp(
            torch.arange(0, dim // 2, 2).float() * (-math.log(img_shape[0] * img_shape[1]) / (dim // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        pe = pe.view(dim, -1)
        return pe.unsqueeze(0)

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
        residuals = residuals.view(N * self.num_clusters, C, H, W)
        pos = self.pe.view(1, C, H, W)
        res_pos = residuals + pos
        residuals = self.pos_fuse(res_pos)
        residuals = residuals.view(N, self.num_clusters, C, -1)

        # weighting
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

        b, n, _, _ = vlad.shape
        vlad = vlad.contiguous()
        vlad = vlad.view(b*n, self.num_clusters, self.dim)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        vlad = self.output_mlp(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization

        vlad = vlad.view(b, n, -1)

        return vlad

