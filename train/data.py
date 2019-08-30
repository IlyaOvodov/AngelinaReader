import os
import random
import json
import PIL
import numpy as np
import albumentations
import albumentations.augmentations.transforms as T
import albumentations.augmentations.functional as albu_f
import torch
import torchvision.transforms.functional as F
import cv2

from DSBI_invest.dsbi import read_txt
import braille_utils.label_tools as lt
import ovotools.pytorch_tools
import local_config


def rect_vflip(b):
    return b[:4] + [lt.label_vflip(b[4]),]

def rect_hflip(b):
    return b[:4] + [lt.label_hflip(b[4]),]


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
    def __init__(self, params, list_file_names, mode, verbose):
        '''
        :param params:  params dict
        :param list_file: file with image files list (relative to local_config.data_path)
        :param mode: augmentation and output mode ('train', 'debug', 'inference')
        :param verbose: if != 0 enables debug print
        '''
        assert mode in {'train', 'debug', 'inference'}
        self.params = params
        self.mode = mode
        self.image_preprocessor = ImagePreprocessor(params, mode)

        self.image_files = []
        self.label_files = []
        for list_file_name in list_file_names:
            list_file = os.path.join(local_config.data_path, list_file_name)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            for fn in files:
                if fn[-1] == '\n':
                    fn = fn[:-1]
                imege_fn, labels_fn = self.filenames_of_item(data_dir, fn)
                if imege_fn:
                    self.image_files.append(imege_fn)
                    self.label_files.append(labels_fn)

        assert len(self.image_files) > 0, list_file

        self.images = [None] * len(self.image_files)
        self.rects = [None] * len(self.image_files)
        self.aug_images = [None] * len(self.image_files)
        self.aug_bboxes = [None] * len(self.image_files)
        self.REPEAT_PROBABILITY = 0.6
        self.get_points = params.data.get('get_points', True)
        self.verbose = verbose

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, item):
        img_fn = self.image_files[item]
        lbl_fn = self.label_files[item]
        img = self.images[item]
        if img is None:
            img = PIL.Image.open(img_fn) #cv2.imread(img_fn) #PIL.Image.open(img_fn)
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            if img.shape[2] == 4:
                img = img[:, :, :3]
            assert len(img.shape) == 3 and img.shape[2] == 3, (img_fn, img.shape)
            self.images[item] = img
        width = img.shape[1]
        height = img.shape[0]
        rects = self.rects[item]
        if rects is None:
            rects = self.read_annotation(lbl_fn, self.get_points, width, height)
            self.rects[item] = rects

        if (self.aug_images[item] is not None) and (random.random() < self.REPEAT_PROBABILITY):
            aug_img = self.aug_images[item]
            aug_bboxes = self.aug_bboxes[item]
        else:
            aug_img, aug_bboxes = self.image_preprocessor.preprocess_and_augment(img, rects)
            self.aug_images[item] = aug_img
            self.aug_bboxes[item] = aug_bboxes

        if self.verbose >= 2:
            print('BrailleDataset: preparing file '+img_fn + '. Total rects: ' + str(len(aug_bboxes)))

        if self.mode == 'train':
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5)
        else:
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5), aug_img

    def filenames_of_item(self, data_dir, fn):
        '''
        finds appropriate image and label full filenames for list item and validates these files exists
        :param data_dir: dir base for filename from list
        :param fn: filename from list
        :return: image filename, label filename or None, None
        '''
        def check_label_ext(imege_fn, ext):
            if not os.path.isfile(imege_fn):
                return None
            lbl_fn = imege_fn.rsplit('.',1)[0]+ext
            if os.path.isfile(lbl_fn):
                return lbl_fn
            return None

        full_fn = os.path.join(data_dir, fn)
        lbl_fn = check_label_ext(full_fn, '.json')
        if lbl_fn:
            return full_fn, lbl_fn
        lbl_fn = check_label_ext(full_fn, '.txt')
        if lbl_fn:
            return full_fn, lbl_fn
        full_fn = full_fn.rsplit('.', 1)[0] + '+recto.jpg'
        lbl_fn = check_label_ext(full_fn, '.txt')
        if lbl_fn:
            return full_fn, lbl_fn
        return None, None

    def read_annotation(self, label_filename, get_points, width, height):
        '''
        :param label_filename:
        :param get_points: if True, returns list of points, label is always 1
        :return: list: [(left,top,right,bottom,label), ...] where coords are (0..1), label is int (0..63)
        '''
        ext = label_filename.rsplit('.', 1)[-1]
        if ext == 'txt':
            return read_DSBI_annotation(self.params, label_filename, get_points, width, height)
        elif ext == 'json':
            return read_LabelMe_annotation(label_filename, get_points)
        else:
            raise ValueError("unsupported label file type: " + ext)


def read_DSBI_annotation(params, label_filename, get_points, width, height):
    _, _, _, cells = read_txt(label_filename, binary_label=True)
    if cells is not None:
        if get_points:
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
                        rects.append([lf / width, tp / height, rt / width, bt / height, 0])  # class is always same
        else:
            rm = params.data.rect_margin
            rects = [[(c.left - rm * (c.right - c.left)) / width,
                      (c.top - rm * (c.right - c.left)) / height,
                      (c.right + rm * (c.right - c.left)) / width,
                      (c.bottom + rm * (c.right - c.left)) / height,
                      lt.label010_to_int(c.label)] for c in cells if c.label != '000000']
    else:
        rects = []
    return rects

def limiting_scaler(source, dest):
    def scale(x):
        return int(min(max(0, x), source-1)) * dest/source
    return scale

def read_LabelMe_annotation(label_filename, get_points):
    with open(label_filename, 'r') as opened_json:
        loaded = json.load(opened_json)
    convert_x = limiting_scaler(loaded["imageWidth"], 1.0)
    convert_y = limiting_scaler(loaded["imageHeight"], 1.0)
    rects = [[convert_x(min(xvals)),
              convert_y(min(yvals)),
              convert_x(max(xvals)),
              convert_y(max(yvals)),
              lt.human_label_to_int(label)
             ] for label, xvals, yvals in
                    ((shape["label"],
                      [coords[0] for coords in shape["points"]],
                      [coords[1] for coords in shape["points"]]
                     ) for shape in loaded["shapes"]
                    )
            ]

    if get_points:
        raise NotImplementedError("read_annotation get_point mode not implemented for LabelMe annotation")
    else:
        return rects


def create_dataloader(params, collate_fn, list_file_names, shuffle, mode = 'train', verbose = 0):
    '''
    :param params:
    :param collate_fn: converts batch from BrailleDataset to format required by model
    :return: train_loader, (val_loader1, val_loader2)
    '''

    '''
    datasets = [BrailleDataset(params, list_file_name=file_name, mode=mode, verbose=verbose)
                for file_name in list_file_names]
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets),
                                               params.data.batch_size,
                                               shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    '''
    dataset = BrailleDataset(params, list_file_names=list_file_names, mode=mode, verbose=verbose)
    loader = torch.utils.data.DataLoader(dataset, params.data.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return loader


if __name__ == '__main__':
    assert rect_hflip( [0,1,2,3, lt.label010_to_int('111000'),] ) == [0,1,2,3, lt.label010_to_int('000111'),]
    assert rect_hflip( [0,1,2,3, lt.label010_to_int('000011'),] ) == [0,1,2,3, lt.label010_to_int('011000'),]
    assert rect_hflip( [0,1,2,3, lt.label010_to_int('001100'),] ) == [0,1,2,3, lt.label010_to_int('100001'),]

    assert rect_vflip( [0,1,2,3, lt.label010_to_int('111100'),] ) == [0,1,2,3, lt.label010_to_int('111001'),]
    assert rect_vflip( [0,1,2,3, lt.label010_to_int('001011'),] ) == [0,1,2,3, lt.label010_to_int('100110'),]

    print('OK')
