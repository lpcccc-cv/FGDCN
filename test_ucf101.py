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
from torch.utils.data import Dataset
from metrics import calculate_psnr, ssim_matlab

class UFC101Dataset(Dataset):
    def __init__(self):
        self.data_root = '/home/lpc/dataset/natural_img/ucf101_interp_ours'
        self.load_data()

    def __len__(self):
        return len(self.data_list)

    def load_data(self):
        dirs = os.listdir(self.data_root)
        data_list = []
        for d in dirs:
            img0 = os.path.join(self.data_root, d, 'frame_00.png')
            img1 = os.path.join(self.data_root, d, 'frame_02.png')
            gt = os.path.join(self.data_root, d, 'frame_01_gt.png')
            data_list.append([img0, img1, gt, d])

        self.data_list = data_list

    def __getitem__(self, index):

        img0 = cv2.imread(self.data_list[index][0])
        img1 = cv2.imread(self.data_list[index][1])
        gt = cv2.imread(self.data_list[index][2])

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]
        gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]

        # im1 = Image.open(self.data_list[index][0]).convert('RGB')
        # im2 = Image.open(self.data_list[index][2]).convert('RGB')
        # im3 = Image.open(self.data_list[index][1]).convert('RGB')

        # im1 = to_tensor(im1)
        # im2 = to_tensor(im2)
        # im3 = to_tensor(im3)

        return torch.stack([img0, img1], dim=0), gt


def Vimeo_PSNR(model, dataloader):
    psnr_total = 0
    ssim_total = 0
    n_samples = 0
    
    for i,data in enumerate(tqdm(dataloader)):
        input_frames, target_frames = data
        input_frames = input_frames.cuda().float()
        target_frames = target_frames.cuda().float()
        b = target_frames.shape[0]
        n_samples += b

        with torch.no_grad():
            pred,_ = model(input_frames)

            psnr_total += calculate_psnr(pred, target_frames)
            ssim_total += ssim_matlab(pred, target_frames)

    avg_psnr = psnr_total / n_samples
    avg_ssmi = ssim_total / n_samples
    print('test_number:', n_samples)
    print(f'Validation PSNR: {avg_psnr}\tSSIM: {avg_ssmi}')
    return avg_psnr, avg_ssmi



### CUDA_VISIBLE_DEVICES=3 python test_vimeo.py
if __name__ == '__main__':
    model_path = './Iter_final.pth'
    model = Ours.FGDCN().cuda()

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    test_data = UFC101Dataset()
    valid_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    valid_psnr, valid_loss = Vimeo_PSNR(model, valid_loader)

    
