import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


class thuman(data.Dataset):
    
    def __init__(self, dataroot, split):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"
        self.dataroot = dataroot
        self.split = split
        self.partial_paths = self._load_data()
        #self.partial_paths, self.complete_paths = self._load_data()
    
    def __getitem__(self, index):
        partial_path = self.partial_paths[index]
        #complete_path = self.complete_paths[index]

        partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)
        #complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)
        complete_pc = self.random_sample(self.read_point_cloud(partial_path), 16384)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.partial_paths)
        #return len(self.complete_paths)

    def _load_data(self):
        partial_paths = []
        #complete_paths = []
        
        for folder in range(526):#526
            folder_path = os.path.join(self.dataroot, f"{folder:04d}")
            for file_name in os.listdir(folder_path):
                if file_name.endswith("_pt_side.ply"):
                    partial_paths.append(os.path.join(folder_path, file_name))
                #elif file_name.endswith(".obj"):
                #    complete_paths.append(os.path.join(folder_path, file_name))
        
        # Shuffle the paths
        random.shuffle(partial_paths)
        #random.shuffle(complete_paths)
        
        # Split according to the defined split ratio
        split_ratio = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
        split_index = int(len(partial_paths) * split_ratio[self.split])
        if self.split == 'train':
            partial_paths = partial_paths[:split_index]
            #complete_paths = complete_paths[:split_index]
        elif self.split == 'valid':
            partial_paths = partial_paths[split_index:2*split_index]
            #complete_paths = complete_paths[split_index:2*split_index]
        else:
            partial_paths = partial_paths[2*split_index:]
            #complete_paths = complete_paths[2*split_index:]
        
        return partial_paths#, complete_paths
    
    def read_point_cloud(self, path):
        '''
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
        '''
        # 读取点云文件
        pc = o3d.io.read_point_cloud(path)
        points = np.array(pc.points, dtype=np.float32)
        
        # 获取点云的坐标范围
        min_coords = np.min(points, axis=1)
        max_coords = np.max(points, axis=1)
        
        # 计算缩放因子和平移量
        scale_factor = 1.0 / np.max(max_coords - min_coords)
        
        # 进行坐标规范化
        scaled_points = points * scale_factor
        
        # 更新点云数据
        pc.points = o3d.utility.Vector3dVector(scaled_points)
        
        # 返回规范化后的点云数组
        return np.array(pc.points, dtype=np.float32)
        
    
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
