import os
import PIL
import numpy as np
import albumentations
import torch
import torchvision.transforms.functional as F

from .dsbi import read_txt
import ovotools.pytorch_tools


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
    def __init__(self, params, data_dir, mode):
        assert mode in {'train', 'test'}
        data_dir_data = os.path.join(data_dir, 'data')
        list_file = os.path.join(data_dir, mode + '.txt')
        with open(list_file, 'r') as f:
            files = f.readlines()
        self.files = [os.path.join(data_dir_data, fn.replace('.jpg\n', '+recto')) for fn in files]
        assert len(self.files) > 0, list_file
        self.albumentations = common_aug(mode, params)
        self.images = [None] * len(self.files)
        self.rects = [None] * len(self.files)

    def __len__(self):
        return len(self.files)
    def __getitem__(self, item):
        fn = self.files[item]
        img = self.images[item]
        if img is None:
            img = PIL.Image.open(fn+'.jpg') #cv2.imread(fn+'.jpg') #PIL.Image.open(fn+'.jpg')
            img = np.asarray(img)
            self.images[item] = img
        rects = self.rects[item]
        if rects is None:
            width = img.shape[1]
            height = img.shape[0]
            _,_,_,cells = read_txt(fn+'.txt', binary_label = True)
            if cells is not None:
                rects = [ (c.left/width, c.top/height, c.right/width, c.bottom/height,
                           self.label_to_int(c.label)) for c in cells if c.label != '000000']
            else:
                rects = []
            self.rects[item] = rects
        #labels = [ self.label_to_int(c.label) for c in cells if c.label != '000000']
        aug_res = self.albumentations(image = img, bboxes = rects)
        aug_img = aug_res['image']
        aug_bboxes = aug_res['bboxes']
        aug_bboxes = [b for b in aug_bboxes if
                      b[0]>0 and b[0]<1 and
                      b[1]>0 and b[1]<1 and
                      b[2]>0 and b[2]<1 and
                      b[3]>0 and b[3]<1]
        return F.to_tensor(aug_img), np.asarray(aug_bboxes)
    def label_to_int(self, label):
        v = [1,2,4,8,16,32]
        r = sum([v[i] for i in range(6) if label[i]=='1'])
        return r


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


def create_dataloaders(params):
    train_dataset = BrailleDataset(params,  r'D:\Programming\Braille\Data\DSBI', mode = 'train')
    val_dataset   = BrailleDataset(params,  r'D:\Programming\Braille\Data\DSBI', mode = 'test')
    val_dataset1 = ovotools.pytorch_tools.DataSubset(val_dataset, list(range(0, len(val_dataset)//2)))
    val_dataset2 = ovotools.pytorch_tools.DataSubset(val_dataset, list(range(len(val_dataset)//2, len(val_dataset))))
    train_loader = torch.utils.data.DataLoader(train_dataset, params.data.batch_size,
                                                      shuffle=True, num_workers=0, collate_fn=detection_collate)
    val_loader1   = torch.utils.data.DataLoader(val_dataset1, params.data.batch_size,
                                                      shuffle=True, num_workers=0, collate_fn=detection_collate)
    val_loader2   = torch.utils.data.DataLoader(val_dataset2, params.data.batch_size,
                                                      shuffle=True, num_workers=0, collate_fn=detection_collate)
    return train_loader, (val_loader1, val_loader2)
