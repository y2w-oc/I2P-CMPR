import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def project_point_cloud_to_range_image(pc_list, H=64, W=900, fov_up=3.0, fov_down=-25.0, device='cpu'):
    """
    :param pc_list: a list of B tensors, each list consists of N point clouds
    :param H: the height of range image
    :param W: the width of range image
    :param fov_up: LiDAR field of view UP
    :param fov_down: LiDAR field of view Down
    :param device: cpu or gpu
    :return: range image , B x N x H x W tensor
    """
    range_list = []
    for sub_list in pc_list:
        temp_list = []
        for pc in sub_list:
            pc = pc[0:3, :].to(device)
            depth = torch.sqrt(torch.sum(pc ** 2, dim=0))
            depth[depth > 50] = 50

            pitch = torch.asin(pc[2, :] / depth) * 180.0 / np.pi
            yaw = torch.atan2(pc[1, :], pc[0, :]) * 180.0 / np.pi

            mask = (pitch < fov_up) & (pitch > fov_down)

            pc = pc[:, mask]
            depth = depth[mask]
            pitch = pitch[mask]
            yaw = yaw[mask]

            fov_total = fov_up - fov_down

            pitch_idx = torch.round((fov_up - pitch) / fov_total * (H - 1))
            pitch_idx = pitch_idx.long()
            yaw_idx = torch.round((- yaw + 180.0) / 360.0 * (W - 1))
            yaw_idx = yaw_idx.long()

            dxyz = torch.cat([depth.unsqueeze(0), pc], dim=0)

            range_image = torch.zeros((4, H, W)).to(device)
            range_image[:, pitch_idx, yaw_idx] = dxyz

            # print(range_image.shape, pitch_idx.shape, yaw_idx.shape, depth.shape, range_image.dtype, depth.dtype)

            # range_img_temp = range_image[0].cpu().numpy()
            # plt.imshow(range_img_temp, cmap='viridis')
            # # plt.colorbar(label='Distance to Object (meters)')
            # plt.title('Positive')
            # plt.xlabel('Horizontal Pixel Index')
            # plt.ylabel('Vertical Pixel Index')
            # plt.show()

            temp_list.append(range_image)

        # plt.subplot(5, 1, 1)
        # range_img_temp = temp_list[0][0].cpu().numpy()
        # plt.imshow(range_img_temp, cmap='viridis')
        # plt.subplot(5, 1, 2)
        # range_img_temp = temp_list[1][0].cpu().numpy()
        # plt.imshow(range_img_temp, cmap='viridis')
        # plt.subplot(5, 1, 3)
        # range_img_temp = temp_list[2][0].cpu().numpy()
        # plt.imshow(range_img_temp, cmap='viridis')
        # plt.subplot(5, 1, 4)
        # range_img_temp = temp_list[3][0].cpu().numpy()
        # plt.imshow(range_img_temp, cmap='viridis')
        # plt.subplot(5, 1, 5)
        # range_img_temp = temp_list[4][0].cpu().numpy()
        # plt.imshow(range_img_temp, cmap='viridis')
        # # plt.colorbar(label='Distance to Object (meters)')
        # plt.title('Positive')
        # plt.xlabel('Horizontal Pixel Index')
        # plt.ylabel('Vertical Pixel Index')
        # plt.show()

        temp_list = torch.stack(temp_list)
        range_list.append(temp_list)

    range_list = torch.stack(range_list)
    return range_list




