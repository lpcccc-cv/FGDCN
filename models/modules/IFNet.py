import torch.nn as nn
import torch.nn.functional as F
import torch
from models.modules.OpticalFlow.flow_lib import flow_to_image
import cv2



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

        self.to_RGB_L1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(96, 3, 3, 1, 1)
        )

        self.to_RGB_L2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(192, 3, 3, 1, 1)
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

        # rec L2 ####
        rec_L2 = self.to_RGB_L2(x_2_5)

        x_1_3 = x_1_3 + self.up[1][0](x_2_3)
        x_0_3 = x_0_3 + self.up[0][0](x_1_3)

        x_1_4 = self.lateral[1][3](x_1_3)
        x_0_4 = self.lateral[0][3](x_0_3)

        x_1_4 = x_1_4 + self.up[1][1](x_2_4)
        x_0_4 = x_0_4 + self.up[0][1](x_1_4)

        x_1_5 = self.lateral[1][4](x_1_4)
        x_0_5 = self.lateral[0][4](x_0_4)

        x_1_5 = x_1_5 + self.up[1][2](x_2_5)

        ## rec L1 ####
        rec_L1 = self.to_RGB_L1(x_1_5)
        #########

        x_0_5 = x_0_5 + self.up[0][2](x_1_5)

        # final synthesis
        output = self.lateral[0][5](x_0_5)
        out = self.to_RGB(output)

        return [out, rec_L1, rec_L2]
        # return out



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


