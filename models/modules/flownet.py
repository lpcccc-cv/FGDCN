import torch.nn as nn
import torch.nn.functional as F
import torch
from models.modules.OpticalFlow.flow_lib import flow_to_image
import cv2

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False)
        return flow

class FlowRefineNetA(nn.Module):
    def __init__(self, context_dim, c=16, r=1, n_iters=4):
        super(FlowRefineNetA, self).__init__()
        corr_dim = c
        flow_dim = c
        motion_dim = c
        hidden_dim = c

        self.n_iters = n_iters
        self.r = r
        self.n_pts = (r * 2 + 1) ** 2

        self.occl_convs = nn.Sequential(nn.Conv2d(2 * context_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, 1, 1, 1, 0),
                                        nn.Sigmoid())
        
        self.occl_convs_new = nn.Sequential(nn.Conv2d(2 * context_dim, hidden_dim, 3, 1, 1),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, 1, 3, 1, 1),
                                        nn.Sigmoid())

        self.corr_convs = nn.Sequential(nn.Conv2d(self.n_pts, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, corr_dim, 1, 1, 0),
                                        nn.PReLU(corr_dim))

        self.flow_convs = nn.Sequential(nn.Conv2d(2, hidden_dim, 3, 1, 1),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                        nn.PReLU(flow_dim))

        self.motion_convs = nn.Sequential(nn.Conv2d(corr_dim + flow_dim, motion_dim, 3, 1, 1),
                                          nn.PReLU(motion_dim))

        self.gru = nn.Sequential(nn.Conv2d(motion_dim + context_dim * 2 + 2, hidden_dim, 3, 1, 1),
                                 nn.PReLU(hidden_dim),
                                 nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                 nn.PReLU(flow_dim), )

        self.flow_head = nn.Sequential(nn.Conv2d(flow_dim, hidden_dim, 3, 1, 1),
                                       nn.PReLU(hidden_dim),
                                       nn.Conv2d(hidden_dim, 2, 3, 1, 1))

    def L2normalize(self, x, dim=1):
        eps = 1e-12
        norm = x ** 2
        norm = norm.sum(dim=dim, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x/norm)

    def forward_once(self, x0, x1, flow0, flow1, mask):
        B, C, H, W = x0.size()

        x0_unfold = F.unfold(x0, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        x1_unfold = F.unfold(x1, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        contents0 = warp(x0_unfold, flow0)
        contents1 = warp(x1_unfold, flow1)

        contents0 = contents0.view(B, C, self.n_pts, H, W)
        contents1 = contents1.view(B, C, self.n_pts, H, W)

        fea0 = contents0[:, :, self.n_pts // 2, :, :]
        fea1 = contents1[:, :, self.n_pts // 2, :, :]

        # get context feature
        occl = self.occl_convs_new(torch.cat([fea0, fea1], dim=1))
        occl = occl+mask
        fea = fea0 * occl + fea1 * (1 - occl)

        # get correlation features
        fea_view = fea.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        contents0 = contents0.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)
        contents1 = contents1.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)

        fea_view = self.L2normalize(fea_view, dim=-1)
        contents0 = self.L2normalize(contents0, dim=-1)
        contents1 = self.L2normalize(contents1, dim=-1)
        corr0 = torch.einsum('bic,bjc->bij', fea_view, contents0)  # (B*H*W, 1, n_pts)
        corr1 = torch.einsum('bic,bjc->bij', fea_view, contents1)
        # corr0 = corr0 / torch.sqrt(torch.tensor(C).float())
        # corr1 = corr1 / torch.sqrt(torch.tensor(C).float())
        corr0 = corr0.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()  # (B, n_pts, H, W)
        corr1 = corr1.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()
        corr0 = self.corr_convs(corr0)  # (B, corr_dim, H, W)
        corr1 = self.corr_convs(corr1)

        # get flow features
        flow0_fea = self.flow_convs(flow0)
        flow1_fea = self.flow_convs(flow1)

        # merge correlation and flow features, get motion features
        motion0 = self.motion_convs(torch.cat([corr0, flow0_fea], dim=1))
        motion1 = self.motion_convs(torch.cat([corr1, flow1_fea], dim=1))

        # update flows
        inp0 = torch.cat([fea, fea0, motion0, flow0], dim=1)
        delta_flow0 = self.flow_head(self.gru(inp0))
        flow0 = flow0 + delta_flow0
        inp1 = torch.cat([fea, fea1, motion1, flow1], dim=1)
        delta_flow1 = self.flow_head(self.gru(inp1))
        flow1 = flow1 + delta_flow1

        return flow0, flow1, occl

    def forward(self, x0, x1, flow0, flow1, mask):
        for i in range(self.n_iters):
            flow0, flow1, occl = self.forward_once(x0, x1, flow0, flow1, mask)

        return torch.cat([flow0, flow1], dim=1), occl

class FlowRefineNet_Multis(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)

        self.rf_block1 = FlowRefineNetA(context_dim=c, c=c, r=1, n_iters=n_iters)
        self.rf_block2 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)
        self.rf_block3 = FlowRefineNetA(context_dim=4 * c, c=4 * c, r=1, n_iters=n_iters)


    def forward(self, x0, x1, flow):
        bs, c, h, w = x0.shape

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4

        # update flow from small scale
        # init mask
        mask = torch.zeros(bs, 1, h//4, w//4).to(flow.device)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow, mask = self.rf_block3(s_3[:bs], s_3[bs:], flow[:, :2], flow[:, 2:4], mask)  # 1/4
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        mask = F.interpolate(mask, scale_factor=2., mode="bilinear", align_corners=False)
        flow, mask = self.rf_block2(s_2[:bs], s_2[bs:], flow[:, :2], flow[:, 2:4], mask)  # 1/2
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        mask = F.interpolate(mask, scale_factor=2., mode="bilinear", align_corners=False)
        flow, mask = self.rf_block1(s_1[:bs], s_1[bs:], flow[:, :2], flow[:, 2:4], mask)  # 1

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, mask, out0, out1, c0, c1
    
    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs
    
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, scale=4, c=180)
        self.block1 = IFBlock(10, scale=2, c=120)
        self.block2 = IFBlock(10, scale=1, c=90)

        self.refine = FlowRefineNet_Multis()

    def forward(self, x):
        ## 
        img1 = x[:, 0, :, :, :]
        img3 = x[:, 1, :, :, :]
        ## 预测光流
        x = torch.cat((img1, img3), 1)
        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)
        flow_list = [F1, F2, F3]

        ## refine flow
        flow, occ_mask, out0, out1, py_fea0, py_fea1 = self.refine(img1, img3, F3)

        warped_img0 = warp(img1, flow[:, :2])
        warped_img1 = warp(img3, flow[:, 2:])

        out = warped_img0*occ_mask+warped_img1*(1-occ_mask)

        # return out
        # return out, flow_list
        return out, flow, out0, out1, flow_list, py_fea0, py_fea1, occ_mask
       

class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.lateral = nn.ModuleList([
            nn.ModuleList([
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
            ]),
            nn.ModuleList([
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
            ]),
            nn.ModuleList([
                LateralBlock(192),
                LateralBlock(192),
                LateralBlock(192),
                LateralBlock(192),
                LateralBlock(192),
            ])
        ])
        self.down = nn.ModuleList([
            nn.ModuleList([
                DownBlock(48, 96),
                DownBlock(48, 96),
                DownBlock(48, 96),
            ]),
            nn.ModuleList([
                DownBlock(96, 192),
                DownBlock(96, 192),
                DownBlock(96, 192),
            ])
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([
                UpBlock(96, 48),
                UpBlock(96, 48),
                UpBlock(96, 48),
            ]),
            nn.ModuleList([
                UpBlock(192, 96),
                UpBlock(192, 96),
                UpBlock(192, 96),
            ])
        ])
        self.compress = nn.ModuleList([
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.Conv2d(192, 192, 3, 1, 1)
        ])
        self.to_RGB = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(48, 3, 3, 1, 1)
        )


    def forward(self, pyramid):
        x_0_0, x_1_0, x_2_0 = pyramid

        # compress dim 32 x 2 -> 32
        x_0_0 = self.compress[0](x_0_0)
        x_1_0 = self.compress[1](x_1_0)
        x_2_0 = self.compress[2](x_2_0)

        # first half: down & lateral
        x_0_1 = self.lateral[0][0](x_0_0)
        x_0_2 = self.lateral[0][1](x_0_1)
        x_0_3 = self.lateral[0][2](x_0_2)

        x_1_0 = x_1_0 + self.down[0][0](x_0_0)
        x_2_0 = x_2_0 + self.down[1][0](x_1_0)

        x_1_1 = self.lateral[1][0](x_1_0)
        x_2_1 = self.lateral[2][0](x_2_0)

        x_1_1 = x_1_1 + self.down[0][1](x_0_1)
        x_2_1 = x_2_1 + self.down[1][1](x_1_1)

        x_1_2 = self.lateral[1][1](x_1_1)
        x_2_2 = self.lateral[2][1](x_2_1)

        x_1_2 = x_1_2 + self.down[0][2](x_0_2)
        x_2_2 = x_2_2 + self.down[1][2](x_1_2)

        x_1_3 = self.lateral[1][2](x_1_2)
        x_2_3 = self.lateral[2][2](x_2_2)

        # second half: up & lateral
        x_2_4 = self.lateral[2][3](x_2_3)
        x_2_5 = self.lateral[2][4](x_2_4)

        x_1_3 = x_1_3 + self.up[1][0](x_2_3)
        x_0_3 = x_0_3 + self.up[0][0](x_1_3)

        x_1_4 = self.lateral[1][3](x_1_3)
        x_0_4 = self.lateral[0][3](x_0_3)

        x_1_4 = x_1_4 + self.up[1][1](x_2_4)
        x_0_4 = x_0_4 + self.up[0][1](x_1_4)

        x_1_5 = self.lateral[1][4](x_1_4)
        x_0_5 = self.lateral[0][4](x_0_4)

        x_1_5 = x_1_5 + self.up[1][2](x_2_5)
        x_0_5 = x_0_5 + self.up[0][2](x_1_5)

        # final synthesis
        output = self.lateral[0][5](x_0_5)
        out = self.to_RGB(output)

        return out

class LateralBlock(nn.Module):
    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.layers(x)


