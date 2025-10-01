import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import argparse
from config import KittiConfiguration
from dataset import KITTIDataset
from models import MultiModalModel
import faiss
import math
from sklearn.decomposition import PCA


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


@torch.no_grad()
def extract_feat_3d(net, dataloader, dataroot, seq):
    print("Extracting 2d3d feature!")
    feat3DFolder = os.path.join(dataroot, seq, "Feat_3D")
    feat2DFolder = os.path.join(dataroot, seq, "Feat_2D")
    if os.path.exists(feat3DFolder):
        pass
    else:
        os.makedirs(feat3DFolder)
    if os.path.exists(feat2DFolder):
        pass
    else:
        os.makedirs(feat2DFolder)

    i = 0
    for data in tqdm(dataloader):
        rgb_imgs = data['imgs'].cuda()
        rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
        feat_2d = net.rgb_desc_inference(rgb_imgs)

        range_imgs = data['ranges'].cuda()
        range_imgs = range_imgs.unsqueeze(1)
        feat_3d = net.range_desc_inference(range_imgs)
        feat_3d = feat_3d[0]

        feat2DPath = os.path.join(feat2DFolder, "%06d.npy"%i)
        feat3DPath = os.path.join(feat3DFolder, "%06d.npy"%i)

        np.save(feat2DPath, feat_2d.cpu().numpy())
        np.save(feat3DPath, feat_3d.cpu().numpy())
        i = i + 1
        # print("Model saved!")
        # time.sleep(100)


def evaluate_2d3d_matching(dataroot='/media/yao/ssd/KITTI/dataset/sequences',
                           seq='00',
                           threshold=10):
    feat3DFolder = os.path.join(dataroot, seq, "Feat_3D")
    feat2DFolder = os.path.join(dataroot, seq, "Feat_2D")

    pose = np.genfromtxt(os.path.join(dataroot.replace("sequences", "poses_semantic"), seq + ".txt"))
    pose = pose[:, [3, 11]]
    inner = 2 * np.matmul(pose, pose.T)
    xx = np.sum(pose ** 2, 1, keepdims=True)
    distance_matrix = xx - inner + xx.T
    distance_matrix = np.sqrt(np.abs(distance_matrix))
    # print("distance_matrix:", distance_matrix.shape)

    feat_3D_files = os.listdir(feat3DFolder)
    feat_3D_files.sort()
    feat_3D_files = [os.path.join(dataroot, seq, "Feat_3D", v) for v in feat_3D_files]

    feat_2D_files = os.listdir(feat2DFolder)
    feat_2D_files.sort()
    feat_2D_files = [os.path.join(dataroot, seq, "Feat_2D", v) for v in feat_2D_files]

    feats_3D = []
    for i in tqdm(range(len(feat_3D_files)), total=len(feat_3D_files)):
        feat_i = np.load(feat_3D_files[i])
        feats_3D.append(feat_i)
    feats_3D = np.vstack(feats_3D)

    d = feats_3D.shape[1]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 1, faiss.METRIC_L2)
    assert not index.is_trained
    index.train(feats_3D)
    assert index.is_trained
    index.add(feats_3D)

    search_num = 50
    successful_count = np.zeros(search_num)
    file_index = open("index.txt", 'w')
    for i in tqdm(range(len(feat_2D_files)), total=len(feat_2D_files)):
        feat_i = np.load(feat_2D_files[i])
        D, I = index.search(feat_i, 1500)
        idx_list = I[0]
        idx_list = idx_list // 30
        old_idx = []
        successful_flag = 0
        for j in range(idx_list.shape[0]):
            min_idx = idx_list[j]

            if min_idx == i:
                continue

            if min_idx in old_idx:
                pass
            else:
                # print(i, j, min_idx)
                if len(old_idx) < search_num:
                    old_idx.append(min_idx)
                    if distance_matrix[i, min_idx] <= threshold and successful_flag == 0:
                        successful_count[len(old_idx)-1:] += 1.0
                        successful_flag += 1
                    if len(old_idx) == search_num:
                        break
        file_index.write(str(old_idx[1]) + "\n")
        # if successful_flag == 0:
        #     file_index.write(str(i) + "  " + str(idx_list[0]) + "\n")
        print("RGB {} Point CLoud {} SearchNum {}".format(i, min_idx, len(old_idx)))
        recall_list = successful_count / (i+1)
        print("Step {} FeatDistance {}  PosDistance {} Recall: {}".format(i, D[0,0], distance_matrix[i, min_idx], recall_list))
    recall_list = recall_list * 100
    np.savetxt("Recall.txt", recall_list, fmt='%.2f')
    file_index.close()


if __name__=='__main__':

    test_sequence = '00'
    matching_threshold = 10

    set_seed(2024)

    data_root = '/media/yao/ssd/KITTI/dataset/sequences'

    device = 'cuda:0'

    test_dataset = KITTIDataset(seqs=[test_sequence],
                                root=data_root,
                                mode='test')

    set_seed(2024)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=False, num_workers=12)

    model = MultiModalModel()
    model = model.to(device)

    print("=============================================================================================>")
    checkpoint_path = "checkpoint/09-02-19-06-04/pretrained.pth"
    print("Checkpoint " + checkpoint_path + " Loading!")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    print("=============================================================================================>")
    print("Start testing!")
    model.eval()

    extract_feat_3d(model, test_loader, data_root, test_sequence)

    evaluate_2d3d_matching(data_root, test_sequence, matching_threshold)
