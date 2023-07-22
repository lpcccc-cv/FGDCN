from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from random import randint
from tqdm import tqdm
import models.modules.Ours as Ours
import utils.util as util
import cv2
import numpy as np
from metrics import calculate_psnr, ssim_matlab
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models.modules.OpticalFlow.flow_lib import flow_to_image
from matplotlib import pyplot as plt


class Vimeo90k_val(data.Dataset):
    def __init__(self, path, is_train=True):
        super(Vimeo90k_val, self).__init__()
        train_list = os.path.join(path, 'tri_trainlist.txt')
        test_list = os.path.join(path, 'tri_testlist.txt')
        self.frame_list = os.path.join(path, 'sequences')
        self.is_train = is_train
        if self.is_train:
            triplet_list = train_list
        else:
            triplet_list = test_list
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
        
        file_name = triplet_path.replace('/', '_')+'.png'

        triplet_path = os.path.join(self.frame_list, triplet_path)
        im1 = os.path.join(triplet_path, 'im1.png')
        im2 = os.path.join(triplet_path, 'im2.png')
        im3 = os.path.join(triplet_path, 'im3.png')

        # read image file
        im1 = cv2.imread(im1)
        im2 = cv2.imread(im2)
        im3 = cv2.imread(im3)

        im1 = torch.from_numpy(im1.astype('float32') / 255.).float().permute(2, 0, 1)[[2, 1, 0],:,:]
        im2 = torch.from_numpy(im2.astype('float32') / 255.).float().permute(2, 0, 1)[[2, 1, 0],:,:]
        im3 = torch.from_numpy(im3.astype('float32') / 255.).float().permute(2, 0, 1)[[2, 1, 0],:,:]

        return torch.stack([im1, im3], dim=0), im2, file_name

def Vimeo_PSNR(model, dataloader):
    psnr_total = 0
    ssim_total = 0
    n_samples = 0
    
    for i,data in enumerate(tqdm(dataloader)):
        input_frames, target_frames, file_name = data
        file_name = file_name[0]
        input_frames = input_frames.cuda().float()
        target_frames = target_frames.cuda().float()
        b = target_frames.shape[0]
        n_samples += b

        with torch.no_grad():
            pred, _  = model(input_frames)
            psnr_total += calculate_psnr(pred, target_frames)
            ssim_total += ssim_matlab(pred, target_frames)

    avg_psnr = psnr_total / n_samples
    avg_ssmi = ssim_total / n_samples
    print('test_number:', n_samples)
    print(f'Validation PSNR: {avg_psnr}\tSSIM: {avg_ssmi}')
    return avg_psnr, avg_ssmi


### CUDA_VISIBLE_DEVICES=2 python test_vimeo.py
if __name__ == '__main__':
    model_path = './Iter_final.pth'
    model = Ours.FGDCN().cuda()

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    test_data = Vimeo90k_val('/home/lpc/dataset/natural_img/vimeo_triplet', is_train=False)
    valid_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    valid_psnr, valid_loss = Vimeo_PSNR(model, valid_loader)

    
