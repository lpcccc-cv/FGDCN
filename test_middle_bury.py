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
import math


class MiddleburyDataset(Dataset):
    def __init__(self):
        self.data_root = '/home/lpc/dataset/natural_img/Middlebury/'
        self.load_data()

    def __len__(self):
        return len(self.data_list)

    def load_data(self):
        name = ['Beanbags', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale',
                'Urban2', 'Urban3', 'Venus', 'Walking']
        data_list = []
        for i in name:
            img0 = os.path.join(self.data_root, 'other-data/{}/frame10.png'.format(i))
            img1 = os.path.join(self.data_root, 'other-data/{}/frame11.png'.format(i))
            gt = os.path.join(self.data_root, 'other-gt-interp/{}/frame10i11.png'.format(i))
            data_list.append([img0, img1, gt, i])
        self.data_list = data_list

    def __getitem__(self, index):
        img0 = cv2.imread(self.data_list[index][0])
        img1 = cv2.imread(self.data_list[index][1])
        gt = cv2.imread(self.data_list[index][2])
        # print('###', gt.shape)

        # pad HR to be mutiple of 64
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
        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

        return torch.stack([img0, img1], dim=0), gt, pad_nums

def Vimeo_PSNR(model, dataloader):
    psnr_total = 0
    ssim_total = 0
    IE_total = 0
    n_samples = 0
    
    for i,data in enumerate(tqdm(dataloader)):
        input_frames, target_frames, pad_nums = data
        pad_t, pad_d, pad_l, pad_r = pad_nums
        input_frames = input_frames.cuda().float()
        target_frames = target_frames.cuda().float()
        b = target_frames.shape[0]
        n_samples += b

        with torch.no_grad():
            pred,_ = model(input_frames)
            # if pad_t != 0 or pad_d != 0 or pad_l != 0 or pad_r != 0:
            #     _, _, h, w = pred.size()
            #     pred = pred[:, :, pad_t:h-pad_d, pad_l:w-pad_r]
            #     target_frames = target_frames[:, :, pad_t:h-pad_d, pad_l:w-pad_r]

            psnr_total += calculate_psnr(pred, target_frames)
            ssim_total += ssim_matlab(pred, target_frames)

            out = np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
            IE = np.abs((out - (target_frames[0]*255.).detach().cpu().numpy().transpose(1, 2, 0))).mean()
            IE_total += IE

            # save reconstructed image
            # for i in range(b):
            #     to_pil_image(pred[i].cpu()).save(os.path.join(cur_valid_path, f'{name[i]}.png'))

    avg_psnr = psnr_total / n_samples
    avg_ssmi = ssim_total / n_samples
    avg_IE = IE_total / n_samples
    print('test_number:', n_samples)
    print(f'Validation PSNR: {avg_psnr}\tSSIM: {avg_ssmi}\tIE: {avg_IE}')
    return avg_psnr, avg_ssmi


### CUDA_VISIBLE_DEVICES=3 python test_vimeo.py
if __name__ == '__main__':
    model_path = './Iter_final.pth'
    model = Ours.FGDCN().cuda()

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    test_data = MiddleburyDataset()
    valid_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    valid_psnr, valid_loss = Vimeo_PSNR(model, valid_loader)

    
