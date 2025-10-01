import numpy as np
import math


class KittiConfiguration:
    """
    The configuration to train on KiTTi dataset
    """
    def __init__(self, dataroot=None):
        print("Operating on KITTI...")
        # <----------- dataset configuration ---------->
        if dataroot is None:
            self.dataset_root = "/media/yao/ssd/KITTI/dataset/sequences"
        else:
            self.dataset_root = dataroot

        # 'train': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
        self.seq_split = {'train': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
                          'test': ['00'],
                          'val': ['00']
                          # 'train++': ['01', '03', '04', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
                          }

        self.pos_threshold = 3
        self.neg_threshold = 20

        self.device = 'cuda:0'

        self.epochs = 48
        self.learning_rate = 3.2768000000000016e-05   # 2.6214400000000015e-05

        self.train_batch_size = 2
        self.val_batch_size = 1
        self.accumulation_step = 1

        self.embed_dim = 64
        self.num_att_head = 8
        self.dropout = 0.1

