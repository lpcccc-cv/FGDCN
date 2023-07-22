from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import torch.utils.data as data
import os
from random import randint
import numpy as np
import random
import cv2


class Vimeo90k(data.Dataset):
    def __init__(self, opt):
        super(Vimeo90k, self).__init__()
        path = '/home/lpc/dataset/natural_img/vimeo_triplet'
        train_list = os.path.join(path, 'tri_trainlist.txt')
        self.frame_list = os.path.join(path, 'sequences')

        triplet_list = train_list

        with open(triplet_list) as triplet_list_file:
            triplet_list = triplet_list_file.readlines()
            triplet_list_file.close()
        self.triplet_list = triplet_list[:-1]

        self.crop_size = opt['GT_size']

    def __len__(self):
        return len(self.triplet_list)
    
    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

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
        img0 = os.path.join(triplet_path, 'im1.png')
        img1 = os.path.join(triplet_path, 'im2.png')
        img2 = os.path.join(triplet_path, 'im3.png')

        # read flow
        flow_path = os.path.join(triplet_path.replace('sequences', 'flows'), 'flow.npy')
        flow_gt = np.load(flow_path).transpose(1, 2, 0) # h,w,c
        # read image file
        img0 = cv2.imread(img0)[:, :, ::-1] # h,w,c, BGR2RGB
        gt = cv2.imread(img1)[:, :, ::-1]
        img1 = cv2.imread(img2)[:, :, ::-1]

        # crop
        img0, gt, img1, flow_gt = self.aug(img0, gt, img1, flow_gt, self.crop_size, self.crop_size)
        
        # data augmentation - random flip / sequence flip
        if random.uniform(0, 1) < 0.5:  # vertical flip
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
            flow_gt = flow_gt[::-1]
            flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
        if random.uniform(0, 1) < 0.5:  # horizontal flip
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
            flow_gt = flow_gt[:, ::-1]
            flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)
        if random.uniform(0, 1) < 0.5:  # reverse time
            tmp = img1
            img1 = img0
            img0 = tmp
            flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)

        flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

        return {'LQs': torch.stack([img0, img1], 0), 'GT': gt.unsqueeze(0), 'flow_gt': flow_gt}
 
