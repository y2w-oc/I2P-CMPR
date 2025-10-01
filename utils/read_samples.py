#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: read sampled range images of KITTI sequences as single input or batch input


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

"""
    read one needed $file_num range image from sequence $file_num.
    Args:
        data_root_folder: dataset root of KITTI.
        file_num: the index of the needed scan (zfill 6).
        seq_num: the sequence in which the needed scan is (zfill 2).
"""
def read_one_need_from_seq_w_rotation(data_root_folder, file_num, seq_num):
    npy_path = data_root_folder + seq_num + "/depth_xyz/" + file_num + ".npy"
    depth_data = np.load(npy_path)

    depth_data_tensor = torch.from_numpy(depth_data).float().cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    x = np.random.randint(0, depth_data_tensor.shape[-1]-1)

    depth_data_tensor = torch.cat([depth_data_tensor[..., -x:], depth_data_tensor[..., :-x]], dim=-1)

    return depth_data_tensor


def read_one_need_from_seq_npy(data_root_folder, file_num, seq_num):
    npy_path = data_root_folder + seq_num + "/depth_xyz/" + file_num + ".npy"
    depth_data = np.load(npy_path)

    depth_data_tensor = torch.from_numpy(depth_data).float().cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    # depth_data_tensor = depth_data_tensor[:, :, 0:42, :]

    return depth_data_tensor


def read_one_need_from_seq(data_root_folder, file_num, seq_num):
    png_path = data_root_folder + seq_num + "/depth_map/" + file_num + ".png"
    depth_data = \
        np.array(cv2.imread(png_path, cv2.IMREAD_GRAYSCALE))

    depth_data_tensor = torch.from_numpy(depth_data).float().cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def read_one_batch_pos_neg_2_thres(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, pos_overlap, neg_overlap):  # without end

    batch_size = 0

    for j in range(len(train_imgf1)):
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> pos_overlap:
                batch_size = batch_size + 1
            elif train_overlap[j] < neg_overlap:
                batch_size = batch_size + 1
            else:
                pass

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 64, 2048))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> pos_overlap:
                pos_num = pos_num + 1
                pos_flag = True
            elif train_overlap[j] < neg_overlap:
                neg_num = neg_num + 1
            else:
                continue

            depth_data_r = \
                np.load(data_root_folder + train_dir2[j] + "/depth_xyz/"+ train_imgf2[j] + ".npy")

            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    # sample_batch = sample_batch[:, :, 0:42, :]
    # print(sample_batch.shape, pos_num, neg_num)
    return sample_batch, sample_truth, pos_num, neg_num


def read_one_batch_pos_neg_npy(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, pos_overlap, neg_overlap):  # without end

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 64, 2048))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> pos_overlap:
                pos_num = pos_num + 1
                pos_flag = True
            elif train_overlap[j] < neg_overlap:
                neg_num = neg_num + 1

            depth_data_r = \
                np.load(data_root_folder + train_dir2[j] + "/depth_xyz/"+ train_imgf2[j] + ".npy")

            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    # sample_batch = sample_batch[:, :, 0:42, :]
    return sample_batch, sample_truth, pos_num, neg_num


"""
    read one batch of positive samples and negative samples with respect to $f1_index in sequence $f1_seq.
    Args:
        data_root_folder: dataset root of KITTI.
        f1_index: the index of the needed scan (zfill 6).
        f1_seq: the sequence in which the needed scan is (zfill 2).
        train_imgf1, train_imgf2, train_dir1, train_dir2: the index dictionary and sequence dictionary following OverlapNet.
        train_overlap: overlaps dictionary following OverlapNet.
        overlap_thresh: 0.3 following OverlapNet.
"""
def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, overlap_thresh):  # without end

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 64, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1

            depth_data_r = \
                np.array(cv2.imread(data_root_folder + train_dir2[j] + "/depth_map/"+ train_imgf2[j] + ".png",
                            cv2.IMREAD_GRAYSCALE))

            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1


    return sample_batch, sample_truth, pos_num, neg_num






