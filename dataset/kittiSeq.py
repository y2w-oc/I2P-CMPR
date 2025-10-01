import os
import sys
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random
import math
import open3d as o3d
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(".")


def range_projection(current_vertex, return_others=False, stop_order=False):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
          current_vertex: raw point clouds  Nx3
        Returns:
          proj_range: projected range image with depth, each pixel contains the corresponding depth
    """
    # hyper parameters
    fov_up = 3.0
    fov_down = -18.0
    proj_H = 48
    proj_W = 900
    max_range = 75

    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    # current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    # depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    mask = (pitch >= fov_down) & (pitch <= fov_up)
    yaw = yaw[mask]
    pitch = pitch[mask]
    depth = depth[mask]

    if return_others:
        current_vertex = current_vertex[mask, :]
    else:
        pass

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]

    range_pixels = np.stack([proj_y, proj_x], axis=0)
    if not stop_order:
        # proj_y = proj_y[order]
        # proj_x = proj_x[order]
        depth = depth[order]
        range_pixels = range_pixels[:, order]
        if return_others:
            current_vertex = current_vertex[order, :]

    proj_range = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)

    proj_range[range_pixels[0, :], range_pixels[1, :]] = depth

    if return_others:
        return proj_range, current_vertex, range_pixels, mask
    else:
        return proj_range


class KITTISEQDataset(Dataset):
    def __init__(self, seqs=['00'],
                 root="/media/yao/ssd/KITTI/dataset/sequences",
                 pos_threshold=3,
                 neg_threshold=20,
                 mode='train'):
        super().__init__()
        self.seqs = seqs
        self.root = root
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.mode = mode

        # data augmentation setting (Velodyne Coordinate system)
        if mode == 'train':
            self.P_Tx_amplitude = 0.5
            self.P_Ty_amplitude = 0.5
            self.P_Tz_amplitude = 0.2
            self.P_Rx_amplitude = 0.0
            self.P_Ry_amplitude = 0.0
            self.P_Rz_amplitude = math.pi
        else:
            self.P_Tx_amplitude = 0.0
            self.P_Ty_amplitude = 0.0
            self.P_Tz_amplitude = 0.0
            self.P_Rx_amplitude = 0.0
            self.P_Ry_amplitude = 0.0
            self.P_Rz_amplitude = 0.0

        self.poses = []
        self.raw_poses = []

        print("----------------------- Loading Poses ---------------------->")
        for seq in seqs:
            pose_path = os.path.join(root.replace('sequences', 'poses_semantic'), seq + '.txt')
            pose = np.genfromtxt(pose_path)
            self.raw_poses.append(pose)
            pose_seq = []
            for i in range(pose.shape[0]):
                pose_seq_i = np.eye(4)
                pose_seq_i[:3, :4] = pose[i].reshape(-1, 4)
                pose_seq.append(pose_seq_i)
            self.poses.append(pose_seq)

        print("----------------------- Loading Calibs ---------------------->")
        self.calibs = []
        for seq in seqs:
            calib_path = os.path.join(root, str(seq), 'calib.txt')
            item_map = {}
            with open(calib_path, 'r') as file:
                for line in file:
                    key, value = line.split(':', 1)
                    if key == 'Tr':
                        item_map['Tr_cam0_vel'] = np.array([float(x) for x in value.split()]).reshape(-1, 4)
                    elif key == 'P2': # only use the left color camera (P2)
                        KT = np.array([float(x) for x in value.split()]).reshape(-1, 4)
                        K = KT[0:3, 0:3]
                        K_inv = np.linalg.inv(K)
                        T = np.matmul(K_inv, KT)
                        item_map['K'] = K
                        item_map['Tr_cam2_cam0'] = T
                        item_map['KT'] = KT
                    else:
                        continue
                temp = np.eye(4)
                temp[:3, :4] = item_map['Tr_cam0_vel']
                item_map['Tr_cam2_vel'] = np.matmul(item_map['Tr_cam2_cam0'], temp)
            self.calibs.append(item_map)

        print("----------------------- Calculating Relative Poses ---------------------->")
        for i in range(len(self.poses)):
            poses_seq = self.poses[i]
            calib_seq = self.calibs[i]

            T_cam_velo = np.eye(4)
            T_cam_velo[:3, :4] = calib_seq['Tr_cam0_vel']
            T_velo_cam = np.linalg.inv(T_cam_velo)

            # load poses
            pose0_inv = np.linalg.inv(poses_seq[0])

            poses_new = []
            for pose in poses_seq:
                poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
            self.poses[i] = np.array(poses_new)

        key = 0
        acc_num = 0
        self.pairs = {}

        print("----------------------- Calculating Positive and Negative Samples ---------------------->")
        for i in tqdm(range(len(self.raw_poses)), desc="Sequences: "):
            pose = self.raw_poses[i][:, [3, 11]]
            height = self.raw_poses[i][:, [7]]
            height_dis = np.abs(height - height.T)
            height_mask = height_dis < 1
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            # id_pos = np.argwhere((dis < pos_threshold))
            # mmmm = (dis < pos_threshold) * height_mask
            # plt.plot(mmmm.sum(axis=1))
            # plt.show()
            id_pos = np.argwhere((dis < pos_threshold) * height_mask)
            id_neg = np.argwhere(dis < neg_threshold)  # (dis > 5) & (dis < 20)

            # for j in tqdm(range(len(pose)), desc="Frames: "):
            #     positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
            #     negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
            #     self.pairs[key] = {
            #         "query_seq": i,
            #         "query_id": j,
            #         "positives": positives.tolist(),
            #         "negatives": negatives.tolist(),
            #     }
            #     key += 1
            # acc_num = acc_num + len(pose)

            for j in tqdm(range(2, len(pose)), desc="Frames: "):
                seq_pos_mask = (id_pos[:, 0] == j) & (id_pos[:, 1] >= 2)
                positives = id_pos[:, 1][seq_pos_mask] + acc_num
                positives = positives - 2 * (i+1)
                seq_neg_mask = (id_neg[:, 0] == j) & (id_neg[:, 1] >= 2)
                negatives = id_neg[:, 1][seq_neg_mask] + acc_num
                negatives = negatives - 2 * (i+1)
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": negatives.tolist(),
                }
                key += 1
            acc_num = acc_num + len(pose)
        print("Preprocessing complete! Total sample number:", len(self.pairs), acc_num)

    def __len__(self):
        return len(self.pairs)

    def load_pcd(self, idx, seq_loading=False):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "velodyne", id + '.bin')
        points = np.fromfile(file, dtype='float32').reshape(-1, 4)

        if seq_loading:
            id_1 = str(query["query_id"] - 1).zfill(6)
            id_2 = str(query["query_id"] - 2).zfill(6)
            file_1 = os.path.join(self.root, seq, "velodyne", id_1 + '.bin')
            file_2 = os.path.join(self.root, seq, "velodyne", id_2 + '.bin')
            points_1 = np.fromfile(file_1, dtype='float32').reshape(-1, 4)
            points_2 = np.fromfile(file_2, dtype='float32').reshape(-1, 4)
            return points[:, 0:3], points_1[:, 0:3], points_2[:, 0:3]
        else:
            return points[:, 0:3]

    def load_rgb_cam2(self, idx):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        id_1 = str(query["query_id"] - 1).zfill(6)
        id_2 = str(query["query_id"] - 2).zfill(6)
        file = os.path.join(self.root, seq, "image_2", id + '.png')
        file_1 = os.path.join(self.root, seq, "image_2", id_1 + '.png')
        file_2 = os.path.join(self.root, seq, "image_2", id_2 + '.png')
        img = cv2.imread(file)
        img_1 = cv2.imread(file_1)
        img_2 = cv2.imread(file_2)
        return img, img_1, img_2

    def get_random_positive_id(self, query_idx, sample_num):
        positives = self.pairs[query_idx]["positives"]
        if len(positives) < sample_num:
            randid = np.random.choice(range(len(positives)), sample_num, replace=True)
        else:
            randid = np.random.choice(range(len(positives)), sample_num, replace=False)
        pos_id = []
        for i in range(sample_num):
            pos_id.append(positives[randid[i]])
        return pos_id

    def get_random_negative_id(self, query_idx, sample_num):
        negatives = self.pairs[query_idx]["negatives"]
        randid = random.sample(range(len(negatives)), sample_num)
        neg_id = []
        for i in range(sample_num):
            neg_id.append(negatives[randid[i]])
        return neg_id

    @staticmethod
    def angles2rotation_matrix(angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        Generate a random transform matrix according to the configuration
        """
        t = [random.uniform(-self.P_Tx_amplitude, self.P_Tx_amplitude),
             random.uniform(-self.P_Ty_amplitude, self.P_Ty_amplitude),
             random.uniform(-self.P_Tz_amplitude, self.P_Tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]
        if self.mode == 'train':
            if random.random() < 1/3:
                t = [0.0, 0.0, 0.0]
                angles = [0.0, 0.0, 0.0]
            else:
                pass
        else:
            pass
        # print(t)
        # print(angles)
        rotation_mat = self.angles2rotation_matrix(angles)
        # R_diff = Rotation.from_matrix(rotation_mat)
        # print(angles, np.abs(R_diff.as_euler('zyx', degrees=False)))
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t
        return P_random

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def crop_image(self, img, K=None):
        '''
        :param img: N x M x 3, K: 3 x 3
        :return: cropped_size x 3, K_New
        '''
        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)), int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)

        if K is not None:
            K = self.camera_matrix_scaling(K, 0.5)

        img_crop_dy = img.shape[0] - 120
        img_crop_dx = np.floor((img.shape[1] - 600) / 2).astype(np.int32)
        # img = img[img_crop_dy:, img_crop_dx:600+img_crop_dx, :]
        img = img[img_crop_dy:, img_crop_dx:600 + img_crop_dx, ::-1]

        if self.mode == 'train':
            img = self.augment_img(img)

        if K is not None:
            K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        if K is not None:
            return img, K
        else:
            return img

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scaled = s * K
        K_scaled[2, 2] = 1
        return K_scaled

    @staticmethod
    def cal_overlap_with_orientations(range_img):
        window = 200
        binary_mask = range_img > 0
        v_sum = binary_mask.sum(axis=0)
        v_sum_circ = np.concatenate((v_sum[800:], v_sum, v_sum[:100]))
        cumsum = np.cumsum(v_sum_circ)
        cumsum = np.insert(cumsum, 0, 0)
        # print(cumsum)
        windows_sums = cumsum[window:-1] - cumsum[:-window-1]
        windows_sums = windows_sums / v_sum.sum()

        # a = np.zeros(900)
        # for i in range(900):
        #     temp = v_sum_circ[i:i+200]
        #     a[i] = temp.sum()
        # a = a/v_sum.sum()
        # if (windows_sums == a).sum() < 900:
        #     print("Fuck!")
        # plt.plot(windows_sums)
        # plt.plot(a)
        # plt.show()
        return windows_sums

    def __getitem__(self, idx):
        temp_seq_id = []
        temp_frame_id = []

        seq_id = self.pairs[idx]["query_seq"]
        frame_id = self.pairs[idx]["query_id"]
        temp_seq_id.append(self.seqs[seq_id])
        temp_frame_id.append(frame_id)

        img_anchor, img_anchor_1, img_anchor_2 = self.load_rgb_cam2(idx)

        pc = self.load_pcd(idx)

        range_anchor, pc, _, _ = range_projection(pc, return_others=True)

        T_cam2_vel = self.calibs[seq_id]["Tr_cam2_vel"]
        K = self.calibs[seq_id]["K"]
        poses = self.poses[seq_id]
        current_pose = poses[frame_id]

        img_anchor, K = self.crop_image(img_anchor, K)
        img_anchor_1 = self.crop_image(img_anchor_1)
        img_anchor_2 = self.crop_image(img_anchor_2)

        # fig, axs = plt.subplots(3)
        # axs[0].imshow(img_anchor)
        # axs[1].imshow(img_anchor_1)
        # axs[2].imshow(img_anchor_2)
        # plt.show()

        if self.mode == 'test':
            img_anchor = img_anchor.copy()
            return {
                'imgs': torch.from_numpy(img_anchor).float(),
                'ranges': torch.from_numpy(range_anchor).float(),
            }
        else:
            pass

        # src_pc = o3d.geometry.PointCloud()
        # src_pc.points = o3d.utility.Vector3dVector(pc)
        # plane_model, inliers = src_pc.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        #
        # non_ground_pc = src_pc.select_by_index(inliers, invert=True)
        # pc = np.array(non_ground_pc.points)

        # o3d.visualization.draw_geometries([non_ground_pc])

        img_anchor_x3 = cv2.resize(img_anchor,
                         (int(round(img_anchor.shape[1]) / 3),
                          int(round((img_anchor.shape[0])) / 3)),
                         interpolation=cv2.INTER_LINEAR)
        # print(img_anchor_x3.shape)

        K = self.camera_matrix_scaling(K, 1/3)

        pc_hom = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))

        pc_in_cam2 = np.matmul(T_cam2_vel, pc_hom.T)
        pc_in_cam2 = np.matmul(K, pc_in_cam2)
        H = img_anchor_x3.shape[0]
        W = img_anchor_x3.shape[1]
        pc_in_cam2[:2, :] = pc_in_cam2[:2, :] / pc_in_cam2[2:3, :]

        pixels = pc_in_cam2[:2, :]
        depths = pc_in_cam2[2, :]
        pixels = np.round(pixels)
        pixels = np.stack([pixels[1, :], pixels[0, :]], axis=0)

        current_mask = (0 <= pixels[0, :]) * (pixels[0, :] < H) * (0 <= pixels[1, :]) * (pixels[1, :] < W) * depths > 0
        rgb_pixels = pixels[:, current_mask]

        # depths = depths[current_mask]
        # projection = np.zeros((H,W), dtype=np.float32)
        # img_fuse = img_anchor_x3.copy()
        # for i in range(depths.shape[0]):
        #     x = int(rgb_pixels[0, i])
        #     y = int(rgb_pixels[1, i])
        #     projection[x, y] = depths[i]
        #     img_fuse[x, y] = 255
        # fig, axs = plt.subplots(2)
        # print(img_anchor.shape)
        # axs[0].imshow(img_anchor)
        # axs[1].imshow(img_fuse)
        # plt.show()

        pc_in_cam2 = pc[current_mask, :]
        # range_in_cam = []
        # range_in_cam_anchor, _, range_pixels, _ = range_projection(pc_in_cam2, return_others=True, stop_order=True)
        # range_in_cam.append(range_in_cam_anchor)
        #
        # fig, axs = plt.subplots(4)
        # axs[0].imshow(img_anchor_x3)
        # axs[1].imshow(projection, cmap='viridis')
        # axs[2].imshow(img_fuse)
        # axs[3].imshow(range_anchor)
        # plt.tight_layout()
        # plt.show()

        pos_id = self.get_random_positive_id(idx, 2)
        # pos_id = [idx+10, idx+15, idx+20]
        neg_id = self.get_random_negative_id(idx, 1)

        overlaps = []
        range_samples = []
        rgb_samples = []

        imgs_pos = []
        range_pos = []
        pc_in_cam2_anchor = np.hstack((pc_in_cam2[:, :3], np.ones((pc_in_cam2.shape[0], 1))))
        for i in range(len(pos_id)):
            idx_temp = pos_id[i]
            # img_temp,_,_ = self.load_rgb_cam2(idx_temp)
            # img_temp = self.crop_image(img_temp)
            # imgs_pos.append(img_temp)

            random_T = self.generate_random_transform()

            seq_id = self.pairs[idx_temp]["query_seq"]
            frame_id = self.pairs[idx_temp]["query_id"]
            temp_seq_id.append(self.seqs[seq_id])
            temp_frame_id.append(frame_id)

            reference_pose = poses[frame_id]

            pc_in_cam2_anchor_world = current_pose.dot(pc_in_cam2_anchor.T).T
            pc_in_cam2_r = np.linalg.inv(reference_pose).dot(pc_in_cam2_anchor_world.T).T
            pc_in_cam2_r = np.matmul(random_T, pc_in_cam2_r.T).T
            range_in_cam_r, _, range_pixels, mask = range_projection(pc_in_cam2_r, return_others=True, stop_order=True)

            overlap_map = self.cal_overlap_with_orientations(range_in_cam_r)
            overlaps.append(overlap_map)

            rgb_pixels_temp = rgb_pixels[:, mask]
            sample_idx = random.sample(range(range_pixels.shape[1]), 512)
            range_pixels_train = range_pixels[:, sample_idx]
            rgb_pixels_train = rgb_pixels_temp[:, sample_idx]

            range_samples.append(range_pixels_train)
            rgb_samples.append(rgb_pixels_train)

            # range_in_cam.append(range_in_cam_r)

            pc_temp, pc_temp_1, pc_temp_2 = self.load_pcd(idx_temp, seq_loading=True)
            pc_temp_hom = np.hstack((pc_temp[:, :3], np.ones((pc_temp.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp = range_projection(pc_temp)

            pc_temp_hom = np.hstack((pc_temp_1[:, :3], np.ones((pc_temp_1.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp_1 = range_projection(pc_temp)

            pc_temp_hom = np.hstack((pc_temp_2[:, :3], np.ones((pc_temp_2.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp_2 = range_projection(pc_temp)

            range_pos.append(range_temp_2)
            range_pos.append(range_temp_1)
            range_pos.append(range_temp)

            print(temp_seq_id)
            print(temp_frame_id)
            print(rgb_pixels_train[:, 0:10] * 3)
            print(range_pixels_train[:, 0:10])
            fig, axs = plt.subplots(5)
            axs[0].imshow(img_anchor)
            axs[1].imshow(range_in_cam_r, cmap='viridis')
            axs[2].imshow(range_temp)
            axs[3].imshow(range_temp_1)
            axs[4].imshow(range_temp_2)
            plt.tight_layout()
            plt.show()
            plt.plot(overlap_map)
            plt.show()

        imgs_neg = []
        range_neg = []
        for i in range(len(neg_id)):
            idx_temp = neg_id[i]
            # img_temp = self.load_rgb_cam2(idx_temp)
            # img_temp = self.crop_image(img_temp)
            # imgs_neg.append(img_temp)

            random_T = self.generate_random_transform()

            seq_id = self.pairs[idx_temp]["query_seq"]
            frame_id = self.pairs[idx_temp]["query_id"]
            temp_seq_id.append(self.seqs[seq_id])
            temp_frame_id.append(frame_id)

            # reference_pose = poses[frame_id]
            #
            # pc_in_cam2_anchor_world = current_pose.dot(pc_in_cam2_anchor.T).T
            # pc_in_cam2_r = np.linalg.inv(reference_pose).dot(pc_in_cam2_anchor_world.T).T
            # range_in_cam_r = range_projection(pc_in_cam2_r)
            # range_in_cam.append(range_in_cam_r)

            pc_temp, pc_temp_1, pc_temp_2 = self.load_pcd(idx_temp, seq_loading=True)
            pc_temp_hom = np.hstack((pc_temp[:, :3], np.ones((pc_temp.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp = range_projection(pc_temp)

            pc_temp_hom = np.hstack((pc_temp_1[:, :3], np.ones((pc_temp_1.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp_1 = range_projection(pc_temp)

            pc_temp_hom = np.hstack((pc_temp_2[:, :3], np.ones((pc_temp_2.shape[0], 1))))
            pc_temp = np.matmul(random_T, pc_temp_hom.T).T
            range_temp_2 = range_projection(pc_temp)

            range_neg.append(range_temp_2)
            range_neg.append(range_temp_1)
            range_neg.append(range_temp)

            # fig, axs = plt.subplots(6)
            # axs[0].imshow(img_anchor)
            # axs[1].imshow(img_anchor_1)
            # axs[2].imshow(img_anchor_2)
            # axs[3].imshow(range_temp)
            # axs[4].imshow(range_temp_1)
            # axs[5].imshow(range_temp_2)
            # plt.show()

        # print(img_anchor.shape)
        # imgs = [img_anchor] + imgs_pos + imgs_neg
        # ranges = [range_anchor] + range_pos + range_neg
        ranges = range_pos + range_neg

        img_anchor = np.stack([img_anchor_2, img_anchor_1, img_anchor], axis=0)
        ranges = np.stack(ranges, axis=0)
        overlaps = np.stack(overlaps, axis=0)
        range_pixels = np.stack(range_samples, axis=0)
        rgb_pixels = np.stack(rgb_samples, axis=0)

        # fig, axs = plt.subplots(6)
        # for i in range(2):
        #     axs[3 * i].imshow(ranges[i], cmap='viridis')
        #     axs[3 * i + 1].imshow(range_in_cam[i], cmap='viridis')
        #     axs[3 * i + 2].imshow(imgs[i], cmap='viridis')
        # print(temp_seq_id)
        # print(temp_frame_id)
        # plt.tight_layout()
        # plt.show()

        # print(imgs.shape, ranges.shape, overlaps.shape, range_pixels.shape, rgb_pixels.shape)
        # print(temp_seq_id)
        # print(temp_frame_id)
        # plt.plot(overlaps[0])
        # plt.plot(overlaps[1])
        # plt.show()

        img_anchor = np.ascontiguousarray(img_anchor)
        ranges = np.ascontiguousarray(ranges)
        overlaps = np.ascontiguousarray(overlaps)
        range_pixels = np.ascontiguousarray(range_pixels)
        rgb_pixels = np.ascontiguousarray(rgb_pixels)

        temp_seq_id = [int(x) for x in temp_seq_id]
        temp_frame_id = [int(x) for x in temp_frame_id]
        temp_seq_id = np.array(temp_seq_id)
        temp_frame_id = np.array(temp_frame_id)

        # print(img_anchor.shape, img_anchor_x3.shape, ranges.shape, overlaps.shape, range_pixels.shape, rgb_pixels.shape)

        return {
            'imgs': torch.from_numpy(img_anchor).float(),
            'ranges': torch.from_numpy(ranges).float(),
            'overlaps': torch.from_numpy(overlaps).float(),
            'range_pixels': torch.from_numpy(range_pixels).long(),
            'rgb_pixels': torch.from_numpy(rgb_pixels).long(),
            'seq': torch.from_numpy(temp_seq_id).long(),
            'frame': torch.from_numpy(temp_frame_id).long()
        }


if __name__ == '__main__':

    dataset = KITTISEQDataset(seqs=['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'], mode='train')
    # data = dataset[16500]
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        print(data['imgs'].shape, data['ranges'].shape, data['overlaps'].shape, data['range_pixels'].shape, data['rgb_pixels'].shape)
