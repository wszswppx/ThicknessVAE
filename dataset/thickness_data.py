# 正确版本

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


class thickness_loader(data.Dataset):
    
    def __init__(self, root_dir, split):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"
        self.root_dir = root_dir
        self.split = split
        self.paths = self._load_data()[0], self._load_data()[1], self._load_data()[2] #, self._load_data()[3]
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.paths[0])

    def __getitem__(self, idx):
        
        depth_F_path, depth_B_path, thickness_path = self.paths[0][idx], self.paths[1][idx], self.paths[2][idx]

        depth_F = Image.open(depth_F_path).convert('L')
        depth_B = Image.open(depth_B_path).convert('L')
        thickness = np.load(thickness_path)
        # rendered = Image.open(rendered_path).convert('RGB')
        # 这里插入对thickness的操作
        avg = np.mean(thickness[thickness != 0])
        thickness[thickness==0] = avg
        mask = np.where(cv2.imread(depth_F_path)[:,:,0] < 254, True, False)
        # thickness[~mask]=0
        
        depth_F =self.transform(depth_F)
        mode_value, count = torch.mode(depth_F)
        depth_F[depth_F==mode_value]=torch.mean(depth_F[depth_F!=mode_value])
        depth_B =self.transform(depth_B)
        mode_value, count = torch.mode(depth_B)
        depth_B[depth_B==mode_value]=torch.mean(depth_B[depth_B!=mode_value])
        # thickness =self.transform(thickness)
        # rendered = self.transform(rendered)

        return depth_F, depth_B, thickness, mask, #rendered
        
    def _load_data(self):
        paths = {'depth_F': [], 'depth_B': [], 'thickness': []}
        for folder in range(526):  # 526
            folder_path = os.path.join(self.root_dir, f"{folder:04d}")
            for file_name in os.listdir(folder_path):
                if file_name.endswith("_F.png"):
                    paths['depth_F'].append(os.path.join(folder_path, file_name))
                elif file_name.endswith("_B_.png"):
                    paths['depth_B'].append(os.path.join(folder_path, file_name))
                elif file_name.endswith("thickness_2.npy"):
                    paths['thickness'].append(os.path.join(folder_path, file_name))

        # Randomize using a common index list
        index_list = list(range(len(paths['depth_F'])))
        random.shuffle(index_list)
        for key in paths:
            paths[key] = [paths[key][i] for i in index_list]

        # Split based on ratio
        split_ratio = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
        split_index = int(len(paths['depth_F']) * split_ratio[self.split])
        if self.split == 'train':
            for key in paths:
                paths[key] = paths[key][:split_index]
        elif self.split == 'valid':
            for key in paths:
                paths[key] = paths[key][split_index:2*split_index]
        else:  # test
            for key in paths:
                paths[key] = paths[key][2*split_index:]

        return paths['depth_F'], paths['depth_B'], paths['thickness']
