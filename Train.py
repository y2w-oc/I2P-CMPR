import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import argparse
from config import KittiConfiguration
from dataset import KITTIDataset
from models import LiDARTransformer, MultiModalModel

from utils.utils import NativeScalerWithGradNormCount
from utils.optimizer import build_optimizer
from models.mamba.mamba_config import get_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def circle_loss_pixel(rgb_feat, range_feat, distance_map, dist_threshold, pos_margin, neg_margin, log_scale=10):
    mask = (distance_map <= dist_threshold).float()

    pos_mask = mask
    neg_mask = 1 - mask

    dists = torch.sqrt(torch.sum((rgb_feat.unsqueeze(-1) - range_feat.unsqueeze(-2)) ** 2, dim=1))

    pos = dists - 1e10 * neg_mask
    pos_weight = (pos - pos_margin).detach()
    pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

    # pos_weight[pos_weight>0]=1.
    # positive_row=torch.sum((pos[:,:num_kpt,:]-pos_margin)*pos_weight[:,:num_kpt,:],dim=-1)/(torch.sum(pos_weight[:,:num_kpt,:],dim=-1)+1e-8)
    # positive_col=torch.sum((pos[:,:,:num_kpt]-pos_margin)*pos_weight[:,:,:num_kpt],dim=-2)/(torch.sum(pos_weight[:,:,:num_kpt],dim=-2)+1e-8)
    lse_positive_row = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-1)
    lse_positive_col = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-2)

    # print(lse_positive_col[:, int(lse_positive_col.shape[-1] / 2):])

    lse_positive_col = lse_positive_col[:, :int(lse_positive_col.shape[-1] / 2)]

    neg = dists + 1e10 * pos_mask
    neg_weight = (neg_margin - neg).detach()
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
    # neg_weight[neg_weight>0]=1.
    # negative_row=torch.sum((neg[:,:num_kpt,:]-neg_margin)*neg_weight[:,:num_kpt,:],dim=-1)/torch.sum(neg_weight[:,:num_kpt,:],dim=-1)
    # negative_col=torch.sum((neg[:,:,:num_kpt]-neg_margin)*neg_weight[:,:,:num_kpt],dim=-2)/torch.sum(neg_weight[:,:,:num_kpt],dim=-2)
    lse_negative_row = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-1)
    lse_negative_col = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-2)

    lse_negative_col = lse_negative_col[:, :int(lse_negative_col.shape[-1] / 2)]

    loss_col = F.softplus(lse_positive_row + lse_negative_row) / log_scale
    loss_row = F.softplus(lse_positive_col + lse_negative_col) / log_scale

    # print(loss_col.shape, loss_row.shape)

    loss = loss_col + loss_row

    return torch.mean(loss), dists


def select_rgb_range_feats(rgb_feat, rgb_pixels, pos_range_feat, neg_range_feat, range_pixels):
    pixel_feat_rgb = []
    distance_map = []
    for i in range(rgb_pixels.shape[0]):
        rgb_feat_i = rgb_feat[i]
        rgb_pixels_i = rgb_pixels[i]
        for j in range(rgb_pixels_i.shape[0]):
            rgb_pixels_ij = rgb_pixels_i[j]
            rgb_temp = rgb_feat_i[:, rgb_pixels_ij[0, :], rgb_pixels_ij[1, :]]
            rgb_pixels_ij = rgb_pixels_ij.float()
            distance_map_ij = torch.norm(rgb_pixels_ij.unsqueeze(-1) - rgb_pixels_ij.unsqueeze(-2), p=2, dim=0)
            pixel_feat_rgb.append(rgb_temp)
            distance_map.append(distance_map_ij)

    pixel_feat_rgb = torch.stack(pixel_feat_rgb)
    distance_map = torch.stack(distance_map)

    distance_map_neg = torch.ones_like(distance_map) * torch.inf
    distance_map = torch.cat([distance_map, distance_map_neg], dim=-1)
    # a = distance_map[0].cpu().data.numpy()
    # a = (a <= 1)
    # plt.imshow(a)
    # plt.show()

    pixel_feat_range_pos = []
    for i in range(range_pixels.shape[0]):
        pos_range_feat_i = pos_range_feat[i]
        range_pixels_i = range_pixels[i]

        for j in range(range_pixels_i.shape[0]):
            range_pixels_ij = range_pixels_i[j]
            pos_range_feat_ij = pos_range_feat_i[j]
            range_temp = pos_range_feat_ij[:, range_pixels_ij[0, :], range_pixels_ij[1, :]]
            pixel_feat_range_pos.append(range_temp)
    pixel_feat_range_pos = torch.stack(pixel_feat_range_pos)

    pixel_feat_range_neg = []
    for i in range(range_pixels.shape[0]):
        range_pixels_i = range_pixels[i]
        neg_range_feat_i0 = neg_range_feat[i, 0]
        f, h, w = neg_range_feat_i0.shape
        neg_range_feat_flatten = neg_range_feat_i0.view(f, -1)
        neg_range_feat_flatten = neg_range_feat_flatten.unsqueeze(-2)

        for j in range(range_pixels_i.shape[0]):
            id = i * range_pixels_i.shape[0] + j
            pixel_feat_rgb_ij = pixel_feat_rgb[id]
            with torch.no_grad():
                dist = torch.norm(pixel_feat_rgb_ij.unsqueeze(-1) - neg_range_feat_flatten, p=2, dim=0)
                neg_sample_ids = torch.argmin(dist, dim=1)
            range_temp = neg_range_feat_flatten[:, 0, neg_sample_ids[:]]
            pixel_feat_range_neg.append(range_temp)
    pixel_feat_range_neg = torch.stack(pixel_feat_range_neg)

    pixel_feat_range = torch.cat([pixel_feat_range_pos, pixel_feat_range_neg], dim=-1)

    # print(pixel_feat_rgb.shape, pixel_feat_range.shape, distance_map.shape)

    return pixel_feat_rgb, pixel_feat_range, distance_map


# def select_rgb_range_feats(rgb_feat, rgb_pixels, pos_range_feat, neg_range_feat, range_pixels):
#     pixel_feat_rgb = []
#     distance_map = []
#     for i in range(rgb_pixels.shape[0]):
#         rgb_feat_i = rgb_feat[i]
#         rgb_pixels_i = rgb_pixels[i]
#         for j in range(rgb_pixels_i.shape[0]):
#             rgb_pixels_ij = rgb_pixels_i[j]
#             rgb_temp = rgb_feat_i[:, rgb_pixels_ij[0, :], rgb_pixels_ij[1, :]]
#             rgb_pixels_ij = rgb_pixels_ij.float()
#             distance_map_ij = torch.norm(rgb_pixels_ij.unsqueeze(-1) - rgb_pixels_ij.unsqueeze(-2), p=2, dim=0)
#             pixel_feat_rgb.append(rgb_temp)
#             distance_map.append(distance_map_ij)
#
#     pixel_feat_rgb = torch.stack(pixel_feat_rgb)
#     distance_map = torch.stack(distance_map)
#
#     distance_map_neg = torch.ones_like(distance_map) * torch.inf
#     distance_map = torch.cat([distance_map, distance_map_neg], dim=-1)
#     # a = distance_map[0].cpu().data.numpy()
#     # a = (a <= 1)
#     # plt.imshow(a)
#     # plt.show()
#
#     pixel_feat_range_pos = []
#     pixel_feat_range_neg = []
#     for i in range(range_pixels.shape[0]):
#         pos_range_feat_i = pos_range_feat[i]
#         range_pixels_i = range_pixels[i]
#
#         neg_range_feat_i0 = neg_range_feat[i, 0]
#
#         for j in range(range_pixels_i.shape[0]):
#             range_pixels_ij = range_pixels_i[j]
#             pos_range_feat_ij = pos_range_feat_i[j]
#             range_temp = pos_range_feat_ij[:, range_pixels_ij[0, :], range_pixels_ij[1, :]]
#             pixel_feat_range_pos.append(range_temp)
#
#             range_temp = neg_range_feat_i0[:, range_pixels_ij[0, :], range_pixels_ij[1, :]]
#             pixel_feat_range_neg.append(range_temp)
#     pixel_feat_range_pos = torch.stack(pixel_feat_range_pos)
#     pixel_feat_range_neg = torch.stack(pixel_feat_range_neg)
#
#     pixel_feat_range = torch.cat([pixel_feat_range_pos, pixel_feat_range_neg], dim=-1)
#
#     return pixel_feat_rgb, pixel_feat_range, distance_map


# def circle_loss_img(rgb_desc, pos_range_desc, neg_range_desc, overlaps, overlap_threshold, pos_margin, neg_margin, log_scale=10):
#     rgb_desc = rgb_desc.unsqueeze(1).unsqueeze(1)
#     pos_dis = torch.norm(rgb_desc - pos_range_desc, p=2, dim=-1)
#     neg_dis = torch.norm(rgb_desc - neg_range_desc, p=2, dim=-1)
#     # neg_dis_x = torch.cat([neg_dis, neg_dis], dim=1)
#     all_dis = torch.cat([pos_dis, neg_dis], dim=-1)
#     all_dis = all_dis.view(-1, all_dis.shape[2])
#
#     sup_zero_overlaps = torch.zeros_like(overlaps)
#     all_overlaps = torch.cat([overlaps, sup_zero_overlaps], dim=-1)
#     all_overlaps = all_overlaps.view(-1, all_overlaps.shape[2])
#
#     non_neg_mask = (all_overlaps >= overlap_threshold[0]).float()
#     non_pos_mask = (all_overlaps <= overlap_threshold[1]).float()
#
#     pos = all_dis - 1e10 * non_pos_mask
#     pos_weight = (pos - pos_margin).detach()
#     pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)
#
#     lse_positive = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-1)
#
#     neg = all_dis + 1e10 * non_neg_mask
#     neg_weight = (neg_margin - neg).detach()
#     neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
#
#     lse_negative = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-1)
#
#     loss_g = F.softplus(lse_positive + lse_negative) / log_scale
#
#     return torch.mean(loss_g)


def circle_loss_img(rgb_desc, pos_range_desc, neg_range_desc, overlaps, overlap_threshold, pos_margin, neg_margin, log_scale=10):
    # print(rgb_desc.shape, pos_range_desc.shape, neg_range_desc.shape, overlaps.shape)
    rgb_desc = rgb_desc.unsqueeze(1).unsqueeze(1)
    pos_dis = torch.norm(rgb_desc - pos_range_desc, p=2, dim=-1)
    neg_dis = torch.norm(rgb_desc - neg_range_desc, p=2, dim=-1)
    neg_dis_x = torch.cat([neg_dis, neg_dis], dim=1)
    all_dis = torch.cat([pos_dis, neg_dis_x], dim=-1)
    all_dis = all_dis.view(-1, all_dis.shape[2])

    sup_zero_overlaps = torch.zeros_like(overlaps)
    all_overlaps = torch.cat([overlaps, sup_zero_overlaps], dim=-1)
    all_overlaps = all_overlaps.view(-1, all_overlaps.shape[2])

    non_neg_mask = (all_overlaps >= overlap_threshold[0]).float()
    non_pos_mask = (all_overlaps <= overlap_threshold[1]).float()

    pos = all_dis - 1e10 * non_pos_mask
    pos_weight = (pos - pos_margin).detach()
    pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

    lse_positive = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-1)

    neg = all_dis + 1e10 * non_neg_mask
    neg_weight = (neg_margin - neg).detach()
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)

    lse_negative = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-1)

    loss_g = F.softplus(lse_positive + lse_negative) / log_scale

    return torch.mean(loss_g)


if __name__ == '__main__':
    config = KittiConfiguration(dataroot='/media/yao/ssd/KITTI/dataset/sequences')

    device = config.device

    train_dataset = KITTIDataset(seqs=config.seq_split['train'],
                                 root=config.dataset_root,
                                 pos_threshold=config.pos_threshold,
                                 neg_threshold=config.neg_threshold,
                                 mode='train')
    val_dataset = KITTIDataset(seqs=config.seq_split['val'],
                               root=config.dataset_root,
                               pos_threshold=config.pos_threshold,
                               neg_threshold=config.neg_threshold,
                               mode='val')

    set_seed(2024)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               drop_last=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.val_batch_size,
                                             shuffle=False,
                                             drop_last=False, num_workers=1)

    model = MultiModalModel()
    model = model.to(device)

    # print("=============================================================================================>")
    # checkpoint_path = "checkpoint/07-04-16-11-53/epoch-28.pth"
    # print("Checkpoint " + checkpoint_path + " Loading!")
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint)

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=config.learning_rate,
    #     betas=(0.9, 0.99),
    #     weight_decay=1e-06,
    # )
    mamba_config = get_config("./models/mamba/vmambav2_tiny_224.yaml")
    optimizer = build_optimizer(mamba_config, model)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    now_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    log_dir = os.path.join('./log', now_time)
    ckpt_dir = os.path.join("./checkpoint", now_time)

    if os.path.exists(ckpt_dir):
        pass
    else:
        os.makedirs(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # save the backup files
    shutil.copytree("./config", ckpt_dir + "/config")
    shutil.copytree("./dataset", ckpt_dir + "/dataset")
    shutil.copytree("./models", ckpt_dir + "/models")
    shutil.copy("Train.py", ckpt_dir)

    loss_scaler = NativeScalerWithGradNormCount()

    global_step = 0
    for epoch in range(config.epochs):
        print("==========================================================================================================================================>")
        # print("Epoch {}: start validating!".format(epoch))
        # model.eval()
        # with torch.no_grad():
        #     val_count = 0
        #     loss_I2P_pixel_sum = 0
        #     loss_I2P_img_sum = 0
        #     loss_I2P_total_sum = 0
        #     for data in tqdm(val_loader):
        #         rgb_imgs = data['imgs'].to(device)
        #         rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
        #
        #         range_imgs = data['ranges'].to(device)
        #         b, n, h, w = range_imgs.shape
        #         range_imgs = range_imgs.view(b * n, h, w)
        #         range_imgs = range_imgs.unsqueeze(1)
        #
        #         rgb_feat, range_feat, rgb_descriptor, range_descriptor = model(rgb_imgs, range_imgs)
        #
        #         # calculate pixel-level circle loss
        #         f = range_feat.shape[1]
        #         pos_num = 2
        #         range_feat = range_feat.view(b, n, f, h, w)
        #         pos_range_feat = range_feat[:, 0:pos_num, ...]
        #         neg_range_feat = range_feat[:, pos_num:, ...]
        #
        #         range_pixels = data['range_pixels'].to(device)
        #         rgb_pixels = data['rgb_pixels'].to(device)
        #         # print(rgb_feat.shape, pos_range_feat.shape, neg_range_feat.shape, rgb_descriptor.shape, pos_range_descriptor.shape,
        #         # neg_range_descriptor.shape, overlaps.shape, range_pixels.shape, rgb_pixels.shape)
        #
        #         pixel_feat_rgb, pixel_feat_range, dis_map = select_rgb_range_feats(rgb_feat, rgb_pixels, pos_range_feat,
        #                                                                            neg_range_feat, range_pixels)
        #
        #         loss_I2P_pixel, _ = circle_loss_pixel(rgb_feat=pixel_feat_rgb, range_feat=pixel_feat_range,
        #                                               distance_map=dis_map,
        #                                               dist_threshold=1, pos_margin=0.1, neg_margin=1.4, log_scale=10)
        #
        #         # calculate img-level circle loss
        #         _, n1, f1 = range_descriptor.shape
        #         range_descriptor = range_descriptor.view(b, n, n1, f1)
        #         pos_range_descriptor = range_descriptor[:, 0:pos_num, ...]
        #         neg_range_descriptor = range_descriptor[:, pos_num:, ...]
        #         overlaps = data['overlaps'].to(device)
        #         op_ids = list(range(0, 900, 10))
        #         overlaps = overlaps[:, :, op_ids]
        #
        #         # print(rgb_descriptor.shape, pos_range_descriptor.shape, neg_range_descriptor.shape, overlaps.shape)
        #         loss_I2P_img = circle_loss_img(rgb_desc=rgb_descriptor, pos_range_desc=pos_range_descriptor,
        #                                        neg_range_desc=neg_range_descriptor, overlaps=overlaps,
        #                                        overlap_threshold=(0.2, 0.6), pos_margin=0.1, neg_margin=1.4,
        #                                        log_scale=10)
        #
        #         # calculate total loss
        #         loss = loss_I2P_pixel + loss_I2P_img
        #         # loss = loss_I2P_pixel
        #
        #         loss_I2P_pixel_sum = loss_I2P_pixel_sum + loss_I2P_pixel
        #         loss_I2P_img_sum = loss_I2P_img_sum + loss_I2P_img
        #         loss_I2P_total_sum = loss_I2P_total_sum + loss
        #         val_count += 1
        #     # print("I2P val loss... Total:{}, Pixel:{} !".format(loss_I2P_total_sum / val_count, loss_I2P_pixel_sum / val_count))
        #     print("I2P val loss... Total:{}, Img:{}, Pixel:{} !".format(loss_I2P_total_sum / val_count, loss_I2P_img_sum / val_count, loss_I2P_pixel_sum / val_count))
        #     print("-------------------------------------------------------------------------------------------")
        #     writer.add_scalar('Val_loss_I2P_pixel', loss_I2P_pixel_sum / val_count, global_step=epoch)
        #     writer.add_scalar('Val_loss_I2P_img', loss_I2P_img_sum / val_count, global_step=epoch)
        #     writer.add_scalar('Val_loss_I2P_total', loss_I2P_total_sum / val_count, global_step=epoch)

        torch.cuda.empty_cache()

        model.train()
        print("Training epoch: {}! Current learning rate: {}".format(epoch, optimizer.param_groups[0]['lr']))
        for data in tqdm(train_loader):
            global_step += 1
            optimizer.zero_grad()
            rgb_imgs = data['imgs'].to(device)
            rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)

            range_imgs = data['ranges'].to(device)
            b, n, h, w = range_imgs.shape
            range_imgs = range_imgs.view(b * n, h, w)
            range_imgs = range_imgs.unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=True):
                rgb_feat, range_feat, rgb_descriptor, range_descriptor = model(rgb_imgs, range_imgs)

            # calculate pixel-level circle loss
            f = range_feat.shape[1]
            pos_num = 2
            range_feat = range_feat.view(b, n, f, h, w)
            pos_range_feat = range_feat[:, 0:pos_num, ...]
            neg_range_feat = range_feat[:, pos_num:, ...]

            range_pixels = data['range_pixels'].to(device)
            rgb_pixels = data['rgb_pixels'].to(device)

            # select pixel-level and view-level features from multiple scenes
            pixel_feat_rgb, pixel_feat_range, dis_map = select_rgb_range_feats(rgb_feat, rgb_pixels, pos_range_feat,
                                                                               neg_range_feat, range_pixels)

            # multi-scene and pixel-level loss
            loss_I2P_pixel, _ = circle_loss_pixel(rgb_feat=pixel_feat_rgb, range_feat=pixel_feat_range, distance_map=dis_map,
                                                  dist_threshold=1, pos_margin=0.1, neg_margin=1.4, log_scale=10)

            _, n1, f1 = range_descriptor.shape
            range_descriptor = range_descriptor.view(b, n, n1, f1)
            pos_range_descriptor = range_descriptor[:, 0:pos_num, ...]
            neg_range_descriptor = range_descriptor[:, pos_num:, ...]
            overlaps = data['overlaps'].to(device)
            op_ids = list(range(0, 900, 10))
            overlaps = overlaps[:, :, op_ids]

            # multi-scene and multi-view loss
            loss_I2P_img = circle_loss_img(rgb_desc=rgb_descriptor, pos_range_desc=pos_range_descriptor,
                                           neg_range_desc=neg_range_descriptor, overlaps=overlaps,
                                           overlap_threshold=(0.2, 0.6), pos_margin=0.4, neg_margin=1.2, log_scale=10)

            # calculate total loss
            loss = loss_I2P_pixel + loss_I2P_img
            # loss = loss_I2P_pixel

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=mamba_config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=True)
            # loss.backward()
            # optimizer.step()

            writer.add_scalar('Train_loss_I2P_pixel', loss_I2P_pixel , global_step=global_step)
            writer.add_scalar('Train_loss_I2P_img', loss_I2P_img, global_step=global_step)
            writer.add_scalar('Train_loss_I2P_total', loss, global_step=global_step)

            # if global_step > 1 and global_step % 2500 == 0:
            #     filename = "step-%d.pth" % (global_step)
            #     save_path = os.path.join(ckpt_dir, filename)
            #     torch.save(model.state_dict(), save_path)

        scheduler.step()
        filename = "epoch-%d.pth" % (epoch)
        save_path = os.path.join(ckpt_dir, filename)
        torch.save(model.state_dict(), save_path)
