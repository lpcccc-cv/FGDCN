'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# try:
#     import mc  # import memcached
# except ImportError:
#     pass

logger = logging.getLogger('base')

class Vimeo7Dataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(Vimeo7Dataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        self.LR_index_list = []
        for i in range(self.LR_N_frames):
            self.LR_index_list.append(i*2)

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        self.LR_resolution = tuple(opt['LR_resolution'])
        self.HR_resolution = tuple(opt['HR_resolution'])
        #### directly load image keys
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Vimeo7_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_GT = list(pickle.load(open(cache_keys, 'rb'))['keys'])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        # self.flow_env = lmdb.open(self.opt['dataroot_flow'], readonly=True, lock=False, readahead=False,
        #                         meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        # print(scale)
        N_frames = self.opt['N_frames']
        GT_size = self.opt['GT_size']
        
        
        key = self.paths_GT[index] # key = self.paths_GT[index]  
        name_a = key[:len(key) - (1 + len(key.split('_')[-1]))] # name_a, name_b = key.split('_')
        name_b = key.split('_')[-1]

        # file_a, file_b = key.split('_')
        # flow_path = os.path.join('/home/lpc/dataset/natural_img/vimeo_triplet/flows', file_a, file_b, 'flow.npy')
        # flow_gt = np.load(flow_path).transpose(1, 2, 0)  # (c,h,w)

        ## 读取flow, 读取格式为:h,w,c
        # flow_gt = util.read_img(self.flow_env, key + '_1', (4, 256, 448))


        if N_frames == 3:
            neighbor_list = [1, 2, 3]
        elif N_frames == 4:
            neighbor_list = [1, 2, 3, 4]
        elif N_frames == 5:
            neighbor_list = [1, 2, 3, 4, 5]
        elif N_frames == 7:
            neighbor_list = [1, 2, 3, 4, 5, 6, 7]


        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
            # flow 反转
            # flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)

        self.LQ_frames_list = neighbor_list

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
                
       #### get LQ images
        LQ_size_tuple = self.LR_resolution if self.LR_input else self.HR_resolution
        img_LQ_l = []
        for v in self.LQ_frames_list:
            if self.data_type == 'mc':
                img_LQ = self._read_img_mc(
                    osp.join(self.LQ_root, name_a, name_b, '/{}.png'.format(v)))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.opt['use_crop']:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                # flow_gt = flow_gt[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])                   
         
            ## augmentation - flip, rotate
            # if random.uniform(0, 1) < 0.5:  # vertical flip
            #     img_LQ_l[0] = img_LQ_l[0][::-1]
            #     img_LQ_l[1] = img_LQ_l[1][::-1]
            #     img_LQ_l[2] = img_LQ_l[2][::-1]
            #     flow_gt = flow_gt[::-1]
            #     flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
            # if random.uniform(0, 1) < 0.5:  # horizontal flip
            #     img_LQ_l[0] = img_LQ_l[0][:, ::-1]
            #     img_LQ_l[1] = img_LQ_l[1][:, ::-1]
            #     img_LQ_l[2] = img_LQ_l[2][:, ::-1]
            #     flow_gt = flow_gt[:, ::-1]
            #     flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)

        
        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs[:, :, :, [2, 1, 0]], (0, 3, 1, 2)))).float()
        # flow_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(flow_gt, (2, 0, 1)))).float()

        time_list = []
        for i in range(5):
            time_list.append(torch.Tensor([i / (5 - 1)]))
        time_Tensors = torch.cat(time_list)
        if self.opt['N_frames']==5:
            time_tensor = time_Tensors[[1, 2, 3]]
        elif self.opt['N_frames']==4:
            time_tensor = time_Tensors[[1, 3]]
        elif self.opt['N_frames']==3:
            time_tensor = time_Tensors[[2]]
        elif self.opt['N_frames']==7:
            # time_tensor = time_Tensors[[1, 2, 3]]
            time_tensor = torch.tensor([1/6, 2/6, 3/6, 4/6, 5/6])
        
        
        if self.opt['use_time'] == True:
            return {'LQs': img_LQs[[0, -1], :, :, :], 'GT': img_LQs[1:-1, :, :, :], 'key': key, 'time': time_tensor}
        
        else:
            return {'LQs': img_LQs[[0, -1], :, :, :], 'GT': img_LQs, 'key': key}
            # return {'LQs': img_LQs, 'GT': img_GTs[1:-1, :, :, :], 'key': key}

    def __len__(self):
        return len(self.paths_GT)
