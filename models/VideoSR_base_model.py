import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss, EPE, VGGPerceptualLoss, Ternary, fftLoss
from torchvision import models
import torchvision
import numpy as np
import cv2
from models.modules.OpticalFlow.liteflow import Network 

logger = logging.getLogger('base')

class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)

        self.opt = opt
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.flownet = Network().to(self.device)
        self.flownet.eval()

        if opt['datasets']['train']['use_time'] == True and opt['time_pth'] != None:
            self.netG.load_state_dict(torch.load(opt['time_pth'])['model'], strict=True)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            self.flownet = DistributedDataParallel(self.flownet, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
            self.flownet = DataParallel(self.flownet)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.criterion_flow = EPE().to(self.device)
            self.vgg_loss = VGGPerceptualLoss().to(self.device)
            self.sensus_loss = Ternary().to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            self.fft_loss = fftLoss().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():
                ## 冻结部分参数
                # if 'flow_net' in k:
                #     v.requires_grad = False
                ## 其余部分更新
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2'])) 
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()
    
    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
            # self.flow_GT = data['flow_gt'].to(self.device)


    # def optimize_parameters(self, step):
    #     self.lambda_flow = 0.01
    #     flow_loss = 0
    #     loss = 0

    #     # 计算flow—gt
    #     flow_10 = self.flownet(self.real_H[:, 1], self.real_H[:, 0])
    #     flow_12 = self.flownet(self.real_H[:, 1], self.real_H[:, 2])
    #     flow_gt = torch.cat([flow_10, flow_12], 1)
    #     ## img-gt
    #     label_img2 = F.interpolate(self.real_H[:, 1, :, :, :], scale_factor=0.5, mode='bilinear')
    #     label_img4 = F.interpolate(self.real_H[:, 1, :, :, :], scale_factor=0.25, mode='bilinear')


        # self.optimizer_G.zero_grad()
        # # forward
        # self.fake_H, out_list, flow_list = self.netG(self.var_L)

        # # flow_loss
        # for level in range(len(flow_list)):
        #     fscale = flow_list[level].size(-1) / flow_gt.size(-1)
        #     flow_gt_resize = F.interpolate(flow_gt, scale_factor=fscale, mode="bilinear",
        #                             align_corners=False) * fscale
        #     flow_loss += self.criterion_flow(flow_list[level][:, :2], flow_gt_resize[:, :2], 1).mean()
        #     flow_loss += self.criterion_flow(flow_list[level][:, 2:4], flow_gt_resize[:, 2:4], 1).mean()
        # flow_loss = flow_loss / 2. * self.lambda_flow
        
        # ## fft loss
        # # l_fft_L0 = self.fft_loss(self.fake_H, self.real_H[:, 1, :, :, :])
        # # l_fft_L1 = self.fft_loss(out_list[0], label_img2)
        # # l_fft_L2 = self.fft_loss(out_list[1], label_img4)
        # # l_fft = (l_fft_L0+l_fft_L1+l_fft_L2)*0.01
        
        # ## sensus loss
        # # sensus = self.sensus_loss(self.fake_H, self.real_H[:, 1, :, :, :])
        # # sensus_loss = self.l1_loss(sensus, torch.zeros(sensus.shape).to(self.device))*2
        
        # ### img_loss
        # img_loss_L0 = self.cri_pix(self.fake_H, self.real_H[:, 1, :, :, :])
        # img_loss_L1 = self.l1_loss(label_img2, out_list[0])
        # img_loss_L2 = self.l1_loss(label_img4, out_list[1])
        # img_loss = img_loss_L0 + img_loss_L1 + img_loss_L2
        
        # loss = img_loss + flow_loss
        # loss.backward()
        # self.optimizer_G.step()
        # # set log
        # self.log_dict['l_flow'] = flow_loss.item()
        # self.log_dict['l_img'] = img_loss.item()
        # # self.log_dict['l_sen'] = sensus_loss.item()
        # # self.log_dict['l_fft'] = l_fft.item()
        
    
    def optimize_parameters(self, step):
        self.lambda_flow = 0.01
        flow_loss = 0
        loss = 0

        # 计算flow—gt
        flow_10 = self.flownet(self.real_H[:, 1], self.real_H[:, 0])
        flow_12 = self.flownet(self.real_H[:, 1], self.real_H[:, 2])
        flow_gt = torch.cat([flow_10, flow_12], 1)

        self.optimizer_G.zero_grad()
        # forward
        self.fake_H, flow_list = self.netG(self.var_L)

        # flow_loss
        for level in range(len(flow_list)):
            fscale = flow_list[level].size(-1) / flow_gt.size(-1)
            flow_gt_resize = F.interpolate(flow_gt, scale_factor=fscale, mode="bilinear",
                                    align_corners=False) * fscale
            flow_loss += self.criterion_flow(flow_list[level][:, :2], flow_gt_resize[:, :2], 1).mean()
            flow_loss += self.criterion_flow(flow_list[level][:, 2:4], flow_gt_resize[:, 2:4], 1).mean()
        flow_loss = flow_loss / 2. * self.lambda_flow

        ## fft loss
        l_fft = self.fft_loss(self.fake_H, self.real_H[:, 1, :, :, :])*0.1
        
        ## sensus loss
        sensus = self.sensus_loss(self.fake_H, self.real_H[:, 1, :, :, :])
        sensus_loss = self.l1_loss(sensus, torch.zeros(sensus.shape).to(self.device))*2
        
        ### img_loss
        img_loss = self.cri_pix(self.fake_H, self.real_H[:, 1, :, :, :]) 

        loss = img_loss + flow_loss + l_fft + sensus_loss
        loss.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_flow'] = flow_loss.item()
        self.log_dict['l_img'] = img_loss.item()
        self.log_dict['l_sen'] = sensus_loss.item()
        self.log_dict['l_fft'] = l_fft.item()
        
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, _ = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach().float().cpu()
        out_dict['restore'] = self.fake_H.detach().float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label, epoch=None):
        if epoch != None:
            self.save_network(self.netG, 'G', iter_label, str(epoch))
        else:
            self.save_network(self.netG, 'G', iter_label)
