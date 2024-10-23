import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ovotools import AttrDict
from data_utils.data import BrailleDataset

# CenterNet project
from models.model import create_model
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid
from utils.image import draw_umich_gaussian, draw_msra_gaussian, draw_dense_reg

NUM_CLASSES = 64

def update_params_with_defailts(opt):
    opt = AttrDict(opt.copy())

    opt['arch'] = opt.get('arch', 'dla_34')
    opt['head_conv'] = opt.get('head_conv', 256 if 'dla' in opt.arch else 64)
    opt['heads'] = opt.get('heads', AttrDict(
        hm = NUM_CLASSES, wh = 2, reg = 2,
    ))
    opt['use_hm1'] = opt.get('use_hm1', False)
    if opt.use_hm1:
        opt['heads']['hm1'] = 1
    
    opt['max_objs'] = 1000
    opt['down_ratio'] = 4
    opt['mse_loss'] = False
    opt['reg_loss'] = 'l1'
    opt['dense_wh'] = False
    opt['norm_wh'] = False
    opt['cat_spec_wh'] = False
    opt['num_stacks'] = 1
    opt['eval_oracle_hm'] = False
    opt['eval_oracle_wh'] = False
    opt['eval_oracle_offset'] = False
    opt['reg_offset'] = True
    opt['hm_weight'] = 1
    opt['wh_weight'] = 0.1
    opt['off_weight'] = 1
    opt['hm1_weight'] = 1
    return opt


def create_model_centernet(params, device):
    opt = update_params_with_defailts(params.model_params)

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = model.to(device)

    dataset_class = CenterNetDataset

    collate_fn = None
    
    loss = CenterNetLoss(opt)
    loss = loss.to(device)

    return model, dataset_class, collate_fn, loss


class CenterNetDataset(BrailleDataset):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.opt = update_params_with_defailts(params.model_params)  # compatibility with CenterNet params 
        self.max_objs = self.opt.max_objs
        
    def __getitem__(self, index):
        img_ten, bboxes, *other = super().__getitem__(index)
        sample_params = other[0]
        calc_cls = sample_params.get('calc_cls', True)
        
        assert len(bboxes) < self.max_objs, f'Too many bboxes: {bbox.shape}'
        num_objs = len(bboxes)

        input_h, input_w = img_ten.shape[-2], img_ten.shape[-1]
        down_ratio = self.opt.down_ratio
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = NUM_CLASSES
        bboxes[:, [0, 2]] *= output_w
        bboxes[:, [1, 3]] *= output_h
        
        if self.opt.arch == 'hourglass':
            pad = []
            def pad_to_2n(x):
                s = 1
                while s<x:
                    s *= 2
                return s-x
            for i in [-1, -2]:
                pad += [0, pad_to_2n(img_ten.shape[i])]
            if max(pad) > 0:
                img_ten = F.pad(img_ten, pad, "constant", 0)
                input_h, input_w = img_ten.shape[-2], img_ten.shape[-1]
                output_h, output_w = input_h // down_ratio, input_w // down_ratio

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
            cls_id = min(int(ann[4]), num_classes -1)  # GVNC pseudolabel weight is ignored
            bbox = ann[:4]
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
        ret['calc_cls'] = calc_cls
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


class CenterNetDecoder:
    def __init__(self, params):
        pass
                
    def get_cls_pred(self, pred):
        raise NotImplemented
        return cls_pred

    def decode(self, pred, size_wh, params, num_classes):
        """
        return boxes, labels, scores
        """
        opt = update_params_with_defailts(params.model_params)
        hm = pred[-1]['hm'].sigmoid_()  # BCHW
        wh = pred[-1]['wh']
        reg = pred[-1]['reg'] if opt.reg_offset else None
        if opt.use_hm1:
            hm1 = pred[-1]['hm1'].sigmoid_()  # BCHW
            dets, inds = ctdet_decode(hm1, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=1000)  # BxNx6: (bboxes, scores, classes)
            class_scores = _transpose_and_gather_feat(hm, inds)
            cls_score, cls_id = class_scores.max(dim=2)
            # dets[:,:,4] *= cls_score
            dets[:,:,5] = cls_id
        else:
            dets, _ = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=1000)  # BxNx6: (bboxes, scores, classes)
        dets = dets[0] # remove batch
        cls_thr = params.inference_params.cls_thresh
        dets = dets[dets[:,4] >= cls_thr]
        boxes, labels, scores = dets[:,:4], dets[:,5].int(), dets[:,4]
        boxes[:, [0,2]] *= size_wh[0]/hm.shape[-1]
        boxes[:, [1,3]] *= size_wh[1]/hm.shape[-2]
        return boxes, labels, scores


class CELoss(nn.Module):
  def __init__(self):
    super(CELoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    pred = pred.view(-1, pred.shape[2])
    target = _transpose_and_gather_feat(target, ind)
    target = target.max(dim=2).indices
    target = target.view(-1)
    mask = mask.view(-1)
    loss = F.cross_entropy(pred * mask.unsqueeze(1), target * mask, #weight, ignore_index,
                           reduction='sum',
                          ) #label_smoothing=0)
    loss = loss / (mask.sum() + 1e-4)
    return loss


###############
# copied from CenterNet project
###############

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        opt = update_params_with_defailts(opt)
        self.opt = opt
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss() # FocalLoss() <-- False
        self.crit_ce = CELoss()
        self.crit_reg = (
            RegL1Loss()  # <--
            if opt.reg_loss == "l1"
            else RegLoss()
            if opt.reg_loss == "sl1"  # ? better (use smooth L1 instead of L1)
            else None
        )
        self.crit_wh = (
            torch.nn.L1Loss(reduction="sum")
            if opt.dense_wh
            else NormRegL1Loss()
            if opt.norm_wh
            else RegWeightedL1Loss()
            if opt.cat_spec_wh
            else self.crit_reg  # RegL1Loss() <--
        )

    def forward(self, outputs, batch):
        """
        Args:
            outputs (_type_):
                output["hm"]: class map: BCHW
                output["wh"]: size(wh) map: B2HW
                output["reg"]: delta(x,y) centers by objects: B2HW
            batch (_type_):
                batch["hm"] - gt gaussed class map: BCHW
                batch["wh"] - gt size(wh) by objects: BM2 (M = max_obj)
                batch["reg"] - gt delta(x,y) centers by objects: BM2 (M = max_obj)
                batch['reg_mask'] - object presence among max_obj: BM (1,1, ..., 1,0, ..., 0)
                batch['ind'] - linear index of obj. centers ion map: BM
                batch['calc_cls'] - use data for class training
        Returns:
            loss, loss_stats
        """
        opt = self.opt
        hm_loss, hm1_loss, wh_loss, off_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                if opt.use_hm1:
                    output["hm1"] = _sigmoid(output["hm1"])  # sigmoid + clamp
                output["hm"] = _sigmoid(output["hm"])  # sigmoid + clamp

            if opt.eval_oracle_hm:  # False
                output["hm"] = batch["hm"]
            if opt.eval_oracle_wh:  # False
                output["wh"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["wh"].detach().cpu().numpy(),
                        batch["ind"].detach().cpu().numpy(),
                        output["wh"].shape[3],
                        output["wh"].shape[2],
                    )
                ).to(opt.device)
            if opt.eval_oracle_offset:  # False
                output["reg"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["reg"].detach().cpu().numpy(),
                        batch["ind"].detach().cpu().numpy(),
                        output["reg"].shape[3],
                        output["reg"].shape[2],
                    )
                ).to(opt.device)

            if opt.use_hm1:
                batch_hm1 = batch["hm"].max(dim=1, keepdim=True).values
                hm1_loss += self.crit(output["hm1"], batch_hm1) / opt.num_stacks
                # hm_loss += self.crit_ce(output["hm"], batch["reg_mask"], batch["ind"], batch["hm"]) / opt.num_stacks
            outpu_hm, batch_hm = output["hm"], batch["hm"]
            if batch['calc_cls'].min() == False:
                outpu_hm, batch_hm = outpu_hm[batch['calc_cls']], batch_hm[batch['calc_cls']]
            hm_loss += self.crit(outpu_hm, batch_hm) / opt.num_stacks if batch['calc_cls'].any() else 0
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch["dense_wh_mask"].sum() + 1e-4
                    wh_loss += (
                        self.crit_wh(
                            output["wh"] * batch["dense_wh_mask"],
                            batch["dense_wh"] * batch["dense_wh_mask"],
                        )
                        / mask_weight
                    ) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += (
                        self.crit_wh(
                            output["wh"],
                            batch["cat_spec_mask"],
                            batch["ind"],
                            batch["cat_spec_wh"],
                        )
                        / opt.num_stacks
                    )
                else:
                    wh_loss += (
                        self.crit_reg(
                            output["wh"], batch["reg_mask"], batch["ind"], batch["wh"]
                        )
                        / opt.num_stacks
                    )

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += (
                    self.crit_reg(
                        output["reg"], batch["reg_mask"], batch["ind"], batch["reg"]
                    )
                    / opt.num_stacks
                )

        loss = (
            opt.hm_weight * hm_loss
            + opt.wh_weight * wh_loss
            + opt.off_weight * off_loss
        )
        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        if opt.use_hm1:
            loss += opt.hm1_weight * hm1_loss
            loss_stats['hm1_loss'] = hm1_loss
        return loss, loss_stats


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds // width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections, inds
