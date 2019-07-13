from ovotools import AttrDict

params = AttrDict(
    model_name = 'RFBNet',
    data = AttrDict(
        batch_size = 100,
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        net_hw = (300,300)
    ),
    prior_box = AttrDict( {
        'feature_maps': [19, 10, 5, 3, 2, 1],
        'min_dim': 300,
        'steps': [16, 32, 64, 100, 150, 300],
        'min_sizes': [45, 90, 135, 180, 225, 270],
        'max_sizes': [90, 135, 180, 225, 270, 315],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
    }),
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=1e-3,# momentum=0.9, weight_decay=5e-4
    ),
    max_epoch = 10000,
)

import torch
from torchvision.transforms.functional import to_tensor
import os
import sys
import time
import PIL
import cv2
import numpy as np
import albumentations

sys.path.append(r'D:\Programming\3rd_party\pytorch_addons\RFBNet')
sys.path.append(r'..')

import DSBI_invest.dsbi
import models.RFB_Net_mobile
import layers.modules
num_classes = 64

net = models.RFB_Net_mobile.build_net('train', params.data.net_hw[0], num_classes=num_classes).cuda()
criterion = layers.modules.MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

priorbox = layers.functions.PriorBox(params.prior_box)
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def common_aug(mode, params):
    '''
    :param mode: 'train', 'test'
    '''
    #aug_params = params.get('augm_params', dict())
    augs_list = []
    #assert mode in {'train', 'test'}
    #if mode == 'train':
        #RandomScaleK = aug_params.get('random_scale', 0.2)
        #augs_list += [albumentations.RandomCrop(int(params.data.net_hw[0] / (1-RandomScaleK))+1,
        #                                        int(params.data.net_hw[1] / (1-RandomScaleK))+1 ),]
        #augs_list += [albumentations.RandomScale(RandomScaleK), ]
    augs_list += [albumentations.RandomCrop(params.data.net_hw[0], params.data.net_hw[1]),]
    augs_list += [albumentations.Normalize(mean=params.data.mean, std=params.data.std), ]
    #if mode == 'train':
    #    augs_list += [albumentations.Blur(),
    #                 albumentations.Rotate(limit=aug_params.get('random_rotate', 5)),
    #                 albumentations.RandomBrightness(),
    #                 albumentations.RandomContrast(),
    #                 ]
    return albumentations.Compose(augs_list, p=1.)



class BrailleDataset:
    def __init__(self, data_dir = r'D:\Programming\Braille\Data\DSBI', mode = 'train'):
        assert mode in {'train', 'test'}
        data_dir_data = os.path.join(data_dir, 'data')
        list_file = os.path.join(data_dir, mode + '.txt')
        with open(list_file, 'r') as f:
            files = f.readlines()
        self.files = [os.path.join(data_dir_data, fn.replace('.jpg\n', '+recto')) for fn in files]
        self.albumentations = common_aug(mode, params)
        self.images = [None] * len(self.files)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, item):
        fn = self.files[item]
        img = self.images[item]
        if img is None:
            img = PIL.Image.open(fn+'.jpg') #cv2.imread(fn+'.jpg') #PIL.Image.open(fn+'.jpg')
            img = np.asarray(img)
            self.images[item] = img
        width = img.shape[1]
        height = img.shape[0]
        _,_,_,cells = DSBI_invest.dsbi.read_txt(fn+'.txt', binary_label = True)
        if cells is not None:
            rects = [ (c.left/width, c.top/height, c.right/width, c.bottom/height,
                       self.label010_to_int(c.label)) for c in cells if c.label != '000000']
        else:
            rects = []
        #labels = [ self.label010_to_int(c.label) for c in cells if c.label != '000000']
        aug_res = self.albumentations(image = img, bboxes = rects)
        aug_img = aug_res['image']
        aug_bboxes = aug_res['bboxes']
        aug_bboxes = [b for b in aug_bboxes if
                      b[0]>0 and b[0]<1 and
                      b[1]>0 and b[1]<1 and
                      b[2]>0 and b[2]<1 and
                      b[3]>0 and b[3]<1]
        return to_tensor(aug_img), np.asarray(aug_bboxes)
    def label010_to_int(self, label):
        v = [1,2,4,8,16,32]
        r = sum([v[i] for i in range(6) if label[i]=='1'])
        return r


dataset = BrailleDataset(mode = 'train')

optimizer = eval(params.optim)(net.parameters(), **params.optim_params)

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    epoch_size = len(dataset) // params.data.batch_size + 1
    max_iter = params.max_epoch * epoch_size

    stepvalues = (100 * epoch_size, 300 * epoch_size, 500 * epoch_size, 1000 * epoch_size, 2000 * epoch_size)
    step_index = 0

    start_iter = 0

    lr = params.optim_params.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(torch.utils.data.DataLoader(dataset, params.data.batch_size,
                                                  shuffle=True, num_workers=0,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        #lr = adjust_learning_rate(optimizer, 0.1, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 1 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      loss_l.item(), loss_c.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (params.optim.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = params.optim.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, (img, rects) in enumerate(batch):
        assert torch.is_tensor(img)
        imgs.append(img)
        assert isinstance(rects, type(np.empty(0)))
        annos = torch.from_numpy(rects).float()
        targets.append(annos)
    return (torch.stack(imgs, 0), targets)



if __name__ == '__main__':
    train()
