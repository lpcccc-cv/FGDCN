import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import to_tensor

import data.util as util

class Val_vimeo_dataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    """

    def __init__(self, opt):
        super(Val_vimeo_dataset).__init__()

        path = '/home/lpc/dataset/natural_img/vimeo_triplet'

        test_list = os.path.join(path, 'tri_test_s.txt')
        triplet_list = test_list
        self.frame_list = os.path.join(path, 'sequences')

        with open(triplet_list) as triplet_list_file:
            triplet_list = triplet_list_file.readlines()
            triplet_list_file.close()
        self.triplet_list = triplet_list[:-1]

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        triplet_path = self.triplet_list[idx]
        if triplet_path[-1:] == '\n':
            triplet_path = triplet_path[:-1]
        try:
            vid_no, seq_no = triplet_path.split('/')
            name = vid_no + '_' + seq_no
        except:
            name = triplet_path
        triplet_path = os.path.join(self.frame_list, triplet_path)
        im1 = os.path.join(triplet_path, 'im1.png')
        im2 = os.path.join(triplet_path, 'im2.png')
        im3 = os.path.join(triplet_path, 'im3.png')

        # read image file
        im1 = Image.open(im1).convert('RGB')
        im2 = Image.open(im2).convert('RGB')
        im3 = Image.open(im3).convert('RGB')

        im1 = to_tensor(im1)
        im2 = to_tensor(im2)
        im3 = to_tensor(im3)

        return {'LQs': torch.stack([im1, im3], dim=0), 'GT': torch.stack([im1, im2, im3], dim=0)}
        # return {'LQs': torch.stack([im1, im2, im3], dim=0), 'GT': im2.unsqueeze(0)}

        