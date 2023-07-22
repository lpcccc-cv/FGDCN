#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
# from models.modules.OpticalFlow.flow_lib import flow_to_image
import cv2
##########################################################

# assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

# # torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

# ##########################################################

# arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
# arguments_strOne = './images/one.png'
# arguments_strTwo = './images/two.png'
# arguments_strOut = './out.flo'

# for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#     if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
#     if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
#     if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
#     if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
# end

##########################################################

class SpyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + 'sintel-final' + '.pytorch', file_name='spynet-' + 'sintel-final').items() })
    # end

    def forward(self, tenOne, tenTwo):

        # assert(tenOne.shape[1] == tenTwo.shape[1])
        # assert(tenOne.shape[2] == tenTwo.shape[2])

        intWidth = tenOne.shape[3]
        intHeight = tenOne.shape[2]

        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        # tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
        # tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=tenOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = []

        tenOne = [ self.netPreprocess(tenPreprocessedOne) ]
        tenTwo = [ self.netPreprocess(tenPreprocessedTwo) ]

        for intLevel in range(5):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        ## tenFlow
        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])
        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
        # end
        tenFlow = torch.nn.functional.interpolate(input=tenFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
        
        return tenFlow


    # end
# end


##########################################################

if __name__ == '__main__':
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    spynet = SpyNet().cuda()
    tenOutput = spynet(tenOne.cuda(),tenTwo.cuda())
    
    flow1 = tenOutput.data.cpu()[0]
    flow1 = flow1.numpy().transpose((1,2,0))
    flow_im1 = flow_to_image(flow1)
    cv2.imwrite('./flow1.png', flow_im1)

    flow2 = tenOutput.data.cpu()[1]
    flow2 = flow2.numpy().transpose((1,2,0))
    flow_im2 = flow_to_image(flow2)
    cv2.imwrite('./flow2.png', flow_im2)

# # end