# This code is modified form the Zooming Slow-Mo https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/models/modules/Sakuya_arch.py
from ast import If
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.flownet import IFNet
import models.modules.OpticalFlow.liteflow as liteflow
from torch.nn.functional import interpolate, grid_sample
from models.modules.IFNet import warp,GridNet
import cv2
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep_our
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf):
        super(PCD_Align, self).__init__()
        nf_l3=96
        nf_l2=48
        nf_l1=24
        groups=8
        dcn_kernel = 3
        dcn_padding = dcn_kernel//2
        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf_l3 * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep_our(nf_l3, nf_l3, kernel_size=dcn_kernel, stride=1, padding=dcn_padding, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf_l2 * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv1_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep_our(nf_l2, nf_l2, kernel_size=dcn_kernel, stride=1, padding=dcn_padding, dilation=1,
                                  deformable_groups=groups)
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf_l1 * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv1_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep_our(nf_l1, nf_l1, kernel_size=dcn_kernel, stride=1, padding=dcn_padding, dilation=1,
                                  deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2, py_fea0, flow_py_0t):

        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv3_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(py_fea0[2], L3_offset, flow_py_0t[2]))
        # L2
        B, C, L2_H, L2_W = fea1[1].size()
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.lrelu(self.L2_dcnpack_1(py_fea0[1], L2_offset, flow_py_0t[1]))

        # L1
        B, C, L1_H, L1_W = fea1[0].size()
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))   
        L1_fea = self.L1_dcnpack_1(py_fea0[0], L1_offset, flow_py_0t[0])

        y.append(L1_fea)
        y.append(L2_fea)
        y.append(L3_fea)
        return y


class FGDCN(nn.Module):
    def __init__(self, nf=64, opt=None):
        super(FGDCN, self).__init__()
        self.opt = opt
        self.nf = nf
        ## DCN
        self.pcd_align = PCD_Align(nf=self.nf)
        ## flow
        self.flow_fea_fusin = GridNet()

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU()

        ### flow
        self.flow_net = IFNet()
        # self.flow_net.load_state_dict(torch.load('/home/lpc/VFI/CDIN/experiments/our_flownet_mask_new/models/Iter_95000.pth'))

    def get_flow(self, x):
        img_mid, pre_flow, warped_py_fea0, warped_py_fea1, flow_list, py_fea0, py_fea1, mask = self.flow_net(x)
        return pre_flow, mask

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs

    def forward(self, x, flow_pre=None, mask_pre=None):
        
        # only for inference, flow not None
        if flow_pre is not None:
            _, _, _, _, _, py_fea0, py_fea1, _ = self.flow_net(x)
            pre_flow = flow_pre
            mask = mask_pre
            flow_list = 0
            ## cal img_mid
            img1 = x[:, 0, :, :, :]
            img3 = x[:, 1, :, :, :]
            warped_img0 = warp(img1, pre_flow[:, :2])
            warped_img1 = warp(img3, pre_flow[:, 2:])
            img_mid = warped_img0*mask+warped_img1*(1-mask)
            ## cal warped_py_fea0, warped_py_fea1
            warped_py_fea0 = self.warp_fea(py_fea0, pre_flow[:, :2])
            warped_py_fea1 = self.warp_fea(py_fea1, pre_flow[:, 2:])

        else:
            img_mid, pre_flow, warped_py_fea0, warped_py_fea1, flow_list, py_fea0, py_fea1, mask = self.flow_net(x)  ## 

        flow_t0 = pre_flow[:, 0:2]
        flow_t1 = pre_flow[:, 2:4]

        flow_t0_list = []
        flow_t1_list = []
        mask_list = []
        flow_t0_list.append(flow_t0)
        flow_t1_list.append(flow_t1)
        mask_list.append(mask)
        flow_t0_py = interpolate(flow_t0, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        flow_t1_py = interpolate(flow_t1, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        mask_2 = interpolate(mask, scale_factor=0.5, mode='bilinear', align_corners=False)
        flow_t0_list.append(flow_t0_py)
        flow_t1_list.append(flow_t1_py)
        mask_list.append(mask_2)    
        flow_t0_py = interpolate(flow_t0, scale_factor=0.25, mode='bilinear', align_corners=False) * 0.25
        flow_t1_py = interpolate(flow_t1, scale_factor=0.25, mode='bilinear', align_corners=False) * 0.25
        mask_3 = interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=False)
        flow_t0_list.append(flow_t0_py)
        flow_t1_list.append(flow_t1_py)
        mask_list.append(mask_3)
       
        ### DCN
        DCN_fea_0 = self.pcd_align(warped_py_fea0, warped_py_fea1, py_fea0, flow_t0_list)
        DCN_fea_1 = self.pcd_align(warped_py_fea1, warped_py_fea0, py_fea1, flow_t1_list)
        
        ## flow
        warped_1_L1 = (DCN_fea_0[0] + warped_py_fea0[0])*mask_list[0]
        warped_1_L2 = (DCN_fea_0[1] + warped_py_fea0[1])*mask_list[1]
        warped_1_L3 = (DCN_fea_0[2] + warped_py_fea0[2])*mask_list[2]

        warped_2_L1 = (DCN_fea_1[0] + warped_py_fea1[0])*(1-mask_list[0])
        warped_2_L2 = (DCN_fea_1[1] + warped_py_fea1[1])*(1-mask_list[1])
        warped_2_L3 = (DCN_fea_1[2] + warped_py_fea1[2])*(1-mask_list[2])

        py_fea = [torch.cat([warped_1_L1, warped_2_L1],1), torch.cat([warped_1_L2, warped_2_L2],1), torch.cat([warped_1_L3, warped_2_L3],1)]

        out = self.flow_fea_fusin(py_fea)
        rec = out[0] + img_mid

        # #
        img_mid_L1 = interpolate(img_mid, scale_factor=0.5, mode='bilinear', align_corners=False) + out[1]
        img_mid_L2 = interpolate(img_mid, scale_factor=0.25, mode='bilinear', align_corners=False) + out[2]
        out_list = [img_mid_L1, img_mid_L2]
             
        # return rec, out_list, flow_list
        return rec, flow_list