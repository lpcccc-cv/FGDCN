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
from torchvision import transforms
import math
import torch.nn.functional as F

class SNUFILM(Dataset):
    def __init__(self, data_root, mode='hard'):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        self.data_root = data_root
        test_fn = os.path.join(data_root, 'SNU-FILM', 'test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        
        print("[%s] Test dataset has %d triplets" %  (mode, len(self.frame_list)))


    def __getitem__(self, index):
        
        # Use self.test_all_images:
        imgpaths = self.frame_list[index]
        file_name = imgpaths[0].split('/')[-2] + '_' + imgpaths[0].split('/')[-1]  + '.png'
        
        img0 = cv2.imread(imgpaths[0].replace('data', self.data_root))
        gt = cv2.imread(imgpaths[1].replace('data', self.data_root))
        img1 = cv2.imread(imgpaths[2].replace('data', self.data_root))

        # pad HR to be mutiple of 16
        h, w, c = gt.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            gt = cv2.copyMakeBorder(gt.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            pad_nums = [pad_t, pad_d, pad_l, pad_r]
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
            pad_nums = [pad_t, pad_d, pad_l, pad_r]
        # print(gt.shape)
        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]
        gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)[[2,1,0],:,:]

        return torch.stack([img0, img1], dim=0), gt, pad_nums, file_name

    def __len__(self):
        return len(self.frame_list)


def Vimeo_PSNR(model, dataloader):
    psnr_total = 0
    ssim_total = 0
    n_samples = 0
    
    for i,data in enumerate(tqdm(dataloader)):
        input_frames, target_frames, pad_nums, file_name = data
        file_name = file_name[0]
        pad_t, pad_d, pad_l, pad_r = pad_nums
        input_frames = input_frames.cuda().float()
        target_frames = target_frames.cuda().float()
        b = target_frames.shape[0]
        n_samples += b

        down_scale = 0.5
        with torch.no_grad():
            ## flow
            if down_scale == 0.5:
                img0 = input_frames[:, 0]
                img1 = input_frames[:, 1]
                img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
                img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
                b, c, h, w = img0_down.size()
                if h % 16 != 0 or w % 16 != 0:
                    h_new = math.ceil(h / 16) * 16
                    w_new = math.ceil(w / 16) * 16
                    img0_new = torch.zeros((b, c, h_new, w_new)).to(target_frames.device).float()
                    img1_new = torch.zeros((b, c, h_new, w_new)).to(target_frames.device).float()
                    img0_new[:, :, :h, :w] = img0_down
                    img1_new[:, :, :h, :w] = img1_down
                    img0_down = img0_new
                    img1_down = img1_new 
                flow_down, mask_down = model.get_flow(torch.stack([img0_down, img1_down], 1))
                if h % 16 != 0 or w % 16 != 0:
                    flow_down = flow_down[:, :, :h, :w]
                    mask_down = mask_down[:, :, :h, :w]
                flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale
                mask = F.interpolate(mask_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False)        
                pred,_ = model(input_frames, flow_pre=flow, mask_pre=mask)
            else:
                pred,_ = model(input_frames)

            if pad_t != 0 or pad_d != 0 or pad_l != 0 or pad_r != 0:
                _, _, h, w = pred.size()
                pred = pred[:, :, pad_t:h-pad_d, pad_l:w-pad_r]
                target_frames = target_frames[:, :, pad_t:h-pad_d, pad_l:w-pad_r]
            
            psnr_total += calculate_psnr(pred, target_frames)
            ssim_total += ssim_matlab(pred, target_frames)

    avg_psnr = psnr_total / n_samples
    avg_ssmi = ssim_total / n_samples
    print('test_number:', n_samples)
    print(f'Validation PSNR: {avg_psnr}\tSSIM: {avg_ssmi}')
    return avg_psnr, avg_ssmi



### CUDA_VISIBLE_DEVICES=1 python test_film.py
if __name__ == '__main__':


    model_path = './Iter_final.pth'
    model = Ours.FGDCN().cuda()

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    data_path = '/home/lpc/dataset/natural_img'
    mode = 'hard'   # ['easy', 'medium', 'hard', 'extreme']
    test_data = SNUFILM(data_root=data_path, mode=mode)
    valid_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    valid_psnr, valid_loss = Vimeo_PSNR(model, valid_loader)