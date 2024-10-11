import math
import numpy as np
import torch
from torch import nn
from ovotools import AttrDict
from data_utils.data import BrailleDataset

# CenterNet project
from models.model import create_model
from trains.ctdet import CtdetLoss
from utils.image import draw_umich_gaussian, draw_msra_gaussian, draw_dense_reg

        
def create_model_centernet(params, device):
    opt = params.model_params

    arch = opt.get('arch', 'dla_34')
    head_conv = opt.get('head_conv', 256 if 'dla' in arch else 64)
    heads = opt.get('heads', AttrDict(
        hm = 64, wh = 2, reg = 2,
    ))
    model = create_model(arch, heads, head_conv)
    model = model.to(device)

    dataset_class = CenterNetDataset

    collate_fn = None
    
    opt = opt.loss_params

    opt.mse_loss = False
    opt.reg_loss = 'l1'
    opt.dense_wh = False
    opt.norm_wh = False
    opt.cat_spec_wh = False
    opt.num_stacks = 1
    opt.eval_oracle_hm = False
    opt.eval_oracle_wh = False
    opt.eval_oracle_offset = False
    opt.reg_offset = True
    opt.hm_weight = 1
    opt.wh_weight = 0.1
    opt.off_weight = 1
    
    loss = CenterNetLoss(opt)
    loss = loss.to(device)

    return model, dataset_class, collate_fn, loss

class CenterNetDataset(BrailleDataset):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params = params
        self.opt = params.model_params.loss_params  # compatibility with CenterNet params 
        self.max_objs = 1000
        
    def __getitem__(self, index):
        img_ten, bboxes, *other = super().__getitem__(index)
        
        assert len(bboxes) < self.max_objs, f'Too manny bboxes: {bbox.shape}'
        num_objs = len(bboxes)

        input_h, input_w = img_ten.shape[-2], img_ten.shape[-1]
        down_ratio = self.params.model_params.get('down_ratio', 4)
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = 64

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                        draw_umich_gaussian

        for k in range(num_objs):
            ann = bboxes[k]
            cls_id = int(ann[4])  # GVNC pseudolabel weight is ignored
            bbox = ann[:4]
            bbox[[0, 2]] *= output_w
            bbox[[1, 3]] *= output_h
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = int((w-1)/2)  # use that h>w for Braille
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
            
        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        return img_ten, ret

class CenterNetLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.loss_module = CtdetLoss(opt)  # to train parameters in loss Module
    
    def forward(self, *kargs, **kwargs):
        loss, loss_stats = self.loss_module.forward(*kargs, **kwargs)
        self.loss_dict = loss_stats
        return loss
        
    def metric(self, key):
        def call(*kargs, **kwargs):
            return self.loss_dict.get(key, torch.zeros(1)[0])
        return call
