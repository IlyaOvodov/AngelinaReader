import os
import random
from collections import defaultdict
import PIL
import numpy as np
import albumentations
import albumentations.augmentations.transforms as T
import albumentations.augmentations.functional as albu_f
import torch
import torchvision.transforms.functional as F
import cv2

from DSBI_invest.dsbi import read_txt
import DSBI_invest.letters as letters
import ovotools.pytorch_tools

def common_aug(mode, params):
    '''
    :param mode: 'train', 'test', 'inference'
    :param params:
    '''
    #aug_params = params.get('augm_params', dict())
    augs_list = []
    assert mode  in {'train', 'debug', 'inference'}
    augs_list.append(albumentations.PadIfNeeded(min_height=params.data.net_hw[0], min_width=params.data.net_hw[1],
                                                border_mode=cv2.BORDER_REPLICATE,
                                                always_apply=True))
    if mode == 'train':
        augs_list.append(albumentations.RandomCrop(height=params.data.net_hw[0], width=params.data.net_hw[1], always_apply=True))
        augs_list.append(T.Rotate(limit=params.augmentation.rotate_limit, border_mode=cv2.BORDER_CONSTANT, always_apply=True))
        # augs_list.append(T.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT)) - can't handle boundboxes
    elif mode == 'debug':
        augs_list.append(albumentations.CenterCrop(height=params.data.net_hw[0], width=params.data.net_hw[1], always_apply=True))
    if mode != 'inference':
        augs_list.append(T.Blur(blur_limit=4))
        augs_list.append(T.RandomBrightnessContrast())
        augs_list.append(T.MotionBlur())
        augs_list.append(T.JpegCompression(quality_lower=30, quality_upper=100))
        #augs_list.append(T.VerticalFlip())
        augs_list.append(T.HorizontalFlip())

    return albumentations.Compose(augs_list, p=1., bbox_params = {'format':'albumentations', 'min_visibility':0.5})


class ImagePreprocessor:
    def __init__(self, params, mode):
        assert mode in {'train', 'debug', 'inference'}
        self.params = params
        self.albumentations = common_aug(mode, params)
        self.get_points = params.data.get('get_points', True)

    def preprocess_and_augment(self, img, rects=[]):
        aug_img = self.random_resize_and_stretch(img,
                                                 new_width_range=self.params.augmentation.img_width_range,
                                                 stretch_limit=self.params.augmentation.stretch_limit)
        aug_res = self.albumentations(image=aug_img, bboxes=rects)
        aug_img = aug_res['image']
        aug_bboxes = aug_res['bboxes']
        aug_bboxes = [b for b in aug_bboxes if
                      b[0] > 0 and b[0] < 1 and
                      b[1] > 0 and b[1] < 1 and
                      b[2] > 0 and b[2] < 1 and
                      b[3] > 0 and b[3] < 1]
        if not self.get_points:
            for t in self.albumentations.transforms:
                if isinstance(t, T.VerticalFlip) and t.was_applied:
                    aug_bboxes = [rect_vflip(b) for b in aug_bboxes]
                if isinstance(t, T.HorizontalFlip) and t.was_applied:
                    aug_bboxes = [rect_hflip(b) for b in aug_bboxes]
        return aug_img, aug_bboxes

    def random_resize_and_stretch(self, img, new_width_range, stretch_limit = 0):
        new_width_range = T.to_tuple(new_width_range)
        stretch_limit = T.to_tuple(stretch_limit, bias=1)
        new_sz = int(random.uniform(new_width_range[0], new_width_range[1]))
        stretch = random.uniform(stretch_limit[0], stretch_limit[1])

        img_max_sz = img.shape[1] #max(img.shape[0]*stretch, img.shape[1]) #img.shape[1]  # GVNC - now it is resizing to max
        new_width = int(img.shape[1]*new_sz/img_max_sz)
        new_width = ((new_width+31)//32)*32
        new_height = int(img.shape[0]*stretch*new_sz/img_max_sz)
        new_height = ((new_height+31)//32)*32

        return albu_f.resize(img, height=new_height, width=new_width, interpolation=cv2.INTER_LINEAR)

    def to_normalized_tensor(self, img):
        '''
        returns image converted to FloatTensor and normalized
        '''
        ten_img = F.to_tensor(img)
        means = ten_img.view(3, -1).mean(dim=1)
        std = ten_img.view(3, -1).std(dim=1)
        ten_img = (ten_img - means.view(-1, 1, 1)) / (3*std.view(-1, 1, 1))
        # decolorize
        ten_img = ten_img.mean(dim=0).expand(3, -1, -1)
        return ten_img



class BrailleDataset:
    '''
    return annotated images as: ( img: Tensor CxHxW, np.array(Nx5 - left (0..1), top, right, bottom, class ) )
    '''
    def __init__(self, params, data_dir, fn_suffix, mode, verbose):
        assert mode in {'train', 'debug', 'inference'}
        self.params = params
        self.mode = mode
        self.image_preprocessor = ImagePreprocessor(params, mode)
        data_dir_data = os.path.join(data_dir, 'data')
        self.files = []
        list_files = ('train', 'test') if mode == 'train' else (mode,)
        for fn in list_files:
            list_file = os.path.join(data_dir, fn + '.txt')
            with open(list_file, 'r') as f:
                files = f.readlines()
            files_list = [os.path.join(data_dir_data, fn.replace('.jpg\n', fn_suffix)) for fn in files]
            assert len(files_list) > 0, list_file
            self.files += files_list
        self.images = [None] * len(self.files)
        self.rects = [None] * len(self.files)
        self.aug_images = [None] * len(self.files)
        self.aug_bboxes = [None] * len(self.files)
        self.REPEAT_PROBABILITY = 0.6
        self.get_points = params.data.get('get_points', True)
        self.verbose = verbose

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
        rects = self.rects[item]
        if rects is None:
            if os.path.exists(fn+'.txt'):
                _,_,_,cells = read_txt(fn+'.txt', binary_label = True)
                if cells is not None:
                    if self.get_points:
                        dy = 0.15
                        dx = 0.3
                        rects = []
                        for cl in cells:
                            w = int((cl.right - cl.left) * dx)
                            h = int((cl.bottom - cl.top) * dy)
                            for i in range(6):
                                if cl.label[i] == '1':
                                    iy = i % 3
                                    ix = i - iy
                                    if ix == 0:
                                        xc = cl.left
                                    else:
                                        xc = cl.right
                                    lf, rt = xc - w, xc + w
                                    if iy == 0:
                                        yc = cl.top
                                    elif iy == 1:
                                        yc = (cl.top + cl.bottom) // 2
                                    else:
                                        yc = cl.bottom
                                    tp, bt = yc - h, yc + h
                                    rects.append( [lf / width, tp / height, rt / width, bt / height, 0] ) # class is always same
                    else:
                        rm = self.params.data.rect_margin
                        rects = [ [(c.left  - rm*(c.right-c.left))/width,
                                   (c.top   - rm*(c.right-c.left))/height,
                                   (c.right + rm*(c.right-c.left))/width,
                                   (c.bottom+ rm*(c.right-c.left))/height,
                                   label_to_int(c.label)] for c in cells if c.label != '000000']
                else:
                    rects = []
            else:
                rects = []
            self.rects[item] = rects
        #labels = [ label_to_int(c.label) for c in cells if c.label != '000000']

        if (self.aug_images[item] is not None) and (random.random() < self.REPEAT_PROBABILITY):
            aug_img = self.aug_images[item]
            aug_bboxes = self.aug_bboxes[item]
        else:
            aug_img, aug_bboxes = self.image_preprocessor.preprocess_and_augment(img, rects)
            self.aug_images[item] = aug_img
            self.aug_bboxes[item] = aug_bboxes

        if self.verbose >= 2:
            print('BrailleDataset: preparing file '+fn + '. Total rects: ' + str(len(aug_bboxes)))

        if self.mode == 'train':
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5)
        else:
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5), aug_img


def rect_vflip(b):
    return b[:4] + [label_vflip(b[4]),]

def rect_hflip(b):
    return b[:4] + [label_hflip(b[4]),]

def label_to_int(label):
    v = [1,2,4,8,16,32]
    r = sum([v[i] for i in range(6) if label[i]=='1'])
    return r

def label_vflip(lbl):
    return ((lbl&(1+8))<<2) + ((lbl&(4+32))>>2) + (lbl&(2+16))

def label_hflip(lbl):
    return ((lbl&(1+2+4))<<3) + ((lbl&(8+16+32))>>3)

def int_to_label(label):
    label = int(label)
    v = [1,2,4,8,16,32]
    r = ''.join([ '1' if label&v[i] else '0' for i in range(6)])
    return r

def int_to_label123(label):
    label = int(label)
    v = [1,2,4,8,16,32]
    r = ''.join([ str(i+1) for i in range(6) if label&v[i]])
    return r

def int_to_letter(label, lang):
    letter_dicts = defaultdict(dict, {
        'SYM': letters.sym_map,
        'NUM': letters.num_map,
        'EN': letters.alpha_map_EN,
        'RU': letters.alpha_map_RU,
    })
    d = letters.sym_map
    d.update(letter_dicts[lang])
    return d.get(int_to_label123(label),'')

def create_dataloaders(params, collate_fn,
                       data_dir=r'D:\Programming.Data\Braille\DSBI',
                       fn_suffix = '+recto',
                       mode = 'train', verbose = 0):
    '''
    :param params:
    :param collate_fn: converts batch from BrailleDataset to format required by model
    :return: train_loader, (val_loader1, val_loader2)
    '''
    base_dataset = BrailleDataset(params, data_dir=data_dir , fn_suffix = fn_suffix, mode = mode, verbose = verbose)
    data_len = len(base_dataset)
    if mode=='train' and data_len>10:
        train_dataset = ovotools.pytorch_tools.DataSubset(base_dataset, list(range(0, data_len*8//10)))
        val_dataset1 = ovotools.pytorch_tools.DataSubset(base_dataset, list(range(data_len*8//10, data_len*9//10)))
        val_dataset2 = ovotools.pytorch_tools.DataSubset(base_dataset, list(range(data_len*9//10, data_len)))
        val_loader1   = torch.utils.data.DataLoader(val_dataset1, params.data.batch_size,
                                                          shuffle=False, num_workers=0, collate_fn=collate_fn)
        val_loader2   = torch.utils.data.DataLoader(val_dataset2, params.data.batch_size,
                                                          shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        train_dataset = base_dataset
        val_loader1 = val_loader2 = None
    train_loader = torch.utils.data.DataLoader(train_dataset, params.data.batch_size,
                                                      shuffle=mode=='train', num_workers=0, collate_fn=collate_fn)
    return train_loader, (val_loader1, val_loader2)


if __name__ == '__main__':
    assert label_to_int('100000') == 1
    assert label_to_int('101000') == 1+4
    assert label_to_int('000001') == 32

    assert rect_hflip( [0,1,2,3, label_to_int('111000'),] ) == [0,1,2,3, label_to_int('000111'),]
    assert rect_hflip( [0,1,2,3, label_to_int('000011'),] ) == [0,1,2,3, label_to_int('011000'),]
    assert rect_hflip( [0,1,2,3, label_to_int('001100'),] ) == [0,1,2,3, label_to_int('100001'),]

    assert rect_vflip( [0,1,2,3, label_to_int('111100'),] ) == [0,1,2,3, label_to_int('111001'),]
    assert rect_vflip( [0,1,2,3, label_to_int('001011'),] ) == [0,1,2,3, label_to_int('100110'),]

    assert int_to_label(label_to_int('001011')) == '001011'
    assert int_to_label123(label_to_int('001011')) == '356'

    assert int_to_letter(label_to_int('110110'),'EN') == 'g'
    assert int_to_letter(label_to_int('110110'),'xx') == ''
    assert int_to_letter(label_to_int('000000'),'EN') == ''


    print('OK')
