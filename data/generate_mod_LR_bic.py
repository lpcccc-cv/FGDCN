import os
import sys
import cv2
import numpy as np

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_util import imresize_np
except ImportError:
    pass


def generate_mod_LR_bic(up_scale, sourcedir, savedir, format='.png', crop=[0,0,0,0]): # crop l r t b
    # # params: upscale factor, input directory, output directory
    # saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = savedir # saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    # saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # if not os.path.isdir(os.path.join(savedir, 'HR')):
    #     os.mkdir(os.path.join(savedir, 'HR'))
    # if not os.path.isdir(os.path.join(savedir, 'LR')):
    #     os.mkdir(os.path.join(savedir, 'LR'))
    # if not os.path.isdir(os.path.join(savedir, 'Bic')):
    #     os.mkdir(os.path.join(savedir, 'Bic'))

    # if not os.path.isdir(saveHRpath):
    #     os.mkdir(saveHRpath)
    # else:
    #     print('It will cover ' + str(saveHRpath))

    # if not os.path.isdir(saveLRpath):
    #     os.mkdir(saveLRpath)
    # else:
    #     print('It will cover ' + str(saveLRpath))

    # if not os.path.isdir(saveBicpath):
        # os.mkdir(saveBicpath)
    # else:
        # print('It will cover ' + str(saveBicpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith(format)]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):

        filename = filepaths[i]
        if os.path.exists(os.path.join(saveLRpath, filename)):
            continue
        # print('No.{} -- Processing {}'.format(i, filename))
        image = cv2.imread(os.path.join(sourcedir, filename))

        image = image[0 + crop[2]: image.shape[0] - crop[3], 0 + crop[0]: image.shape[1] - crop[1], :]


        width = int(np.floor(image.shape[1] / up_scale))
        height = int(np.floor(image.shape[0] / up_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:up_scale * height, 0:up_scale * width, :]
        else:
            image_HR = image[0:up_scale * height, 0:up_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # # bic
        # image_Bic = imresize_np(image_LR, up_scale, True)

        # cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        # cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)


if __name__ == "__main__":
    generate_mod_LR_bic(4, '/home/lpc/dataset/VSR/vimeo_triplet_small/sequences/test', '/home/lpc/dataset/VSR/vimeo_triplet_small/sequences_LR')
