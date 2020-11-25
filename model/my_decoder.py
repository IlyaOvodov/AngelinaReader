'''Encode object boxes and labels.'''
import math
from typing import List, Tuple, Union
from torch import Tensor
import torch
import pytorch_retinanet.encoder

from model.utils import meshgrid, box_nms_fast, box_nms

USE_FAST_DECODER = False

class DataEncoder(torch.nn.Module):
    def __init__(self,
                 fpn_skip_layers=0,
                 anchor_areas = [32*32.],
                 aspect_ratios = [1 / 1.], # width/height
                 scale_ratios = [1.],
                 iuo_fit_thr = 0.5, # if iou > iuo_fit_thr => rect fits anchor
                 iuo_nofit_thr = 0.4, # if iou < iuo_nofit_thr => anchor has no fit rect
                 ):
        # type: (List[float], List[float], List[float], float, float) -> None
        super(DataEncoder, self).__init__()
        self.anchor_areas = anchor_areas  # p3 -> p7
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.anchor_wh = self._get_anchor_wh()
        self.iuo_fit_thr = iuo_fit_thr
        self.iuo_nofit_thr = iuo_nofit_thr
        self.input_size = torch.tensor(0)
        self.fpn_skip_layers = fpn_skip_layers

    def forward(self):
        pass

    def num_layers(self):
        '''
        :return: num_layers to be passed to RetinaNet
        '''
        return len(self.anchor_areas)

    def num_anchors(self):
        '''
        :return: num_anchors to be passed to RetinaNet
        '''
        return len(self.aspect_ratios)*len(self.scale_ratios)

    def _get_anchor_wh(self):
        # type: ()->Tensor
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh: List[List[float]] = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        r = torch.tensor(anchor_wh, dtype=torch.float32).view(num_fms, -1, 2)
        return r

    def _get_anchor_boxes(self, input_size, device):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        if not torch.equal(input_size, self.input_size):
            self.input_size = input_size
            num_fms = len(self.anchor_areas)
            assert num_fms == 1
            fm_sizes = [(input_size/math.pow(2.,i + 3 + self.fpn_skip_layers)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes
            num_anchors = self.num_anchors()
            assert num_anchors == 1

            fm_size = fm_sizes[0]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h,fm_w,2)
            wh = self.anchor_wh[0].view(1,1,2).expand(fm_h,fm_w, 2)
            box = torch.cat([xy,wh], 2)  # fm_h,fm_w,4 -> [x,y,w,h]
            self.anchor_boxes = box.to(device)
        return self.anchor_boxes

    @torch.jit.unused
    def encode(self, boxes, labels, input_size):
        # type: (Tensor, Tensor, Tuple[int, int])->Tuple[Tensor, Tensor, Tensor]
        raise NotImplementedError

    def decode(self, loc_preds, cls_preds, input_size, cls_thresh = 0.5, nms_thresh = 0.5, num_classes: List[int] = []):
        # type: (Tensor, Tensor, Tuple[int, int], float, float)->Tuple[Tensor, Tensor, Tensor]
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''

        assert loc_preds.device == cls_preds.device
        assert not isinstance(input_size, int)
        input_size = torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size, loc_preds.device)
        loc_preds = loc_preds.view(anchor_boxes.shape[0], anchor_boxes.shape[1], 4)
        cls_preds = cls_preds.view(anchor_boxes.shape[0], anchor_boxes.shape[1], cls_preds.shape[1])

        loc_xy = loc_preds[:, :, :2]
        loc_wh = loc_preds[:, :, 2:]

        xy = loc_xy * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2]
        wh = loc_wh.exp() * anchor_boxes[:, :, 2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 2)  # [fms_h, fms_w, (xyxy)]

        if len(num_classes) <= 1:
            score, labels = cls_preds.sigmoid().max(2)          # [fms_h, fms_w, ]
        else:
            raise NotImplementedError
            # score, _ = cls_preds.sigmoid().max(2)  # [fms_h, fms_w, ]
            # # TODO below, not implemented
            # labels = torch.zeros((len(cls_preds), len(num_classes)), dtype=torch.int, device=cls_preds.device)
            # pos = 0
            # for i, n in enumerate(num_classes):
            #     score_i, labels[:, i] = cls_preds[:, pos:pos+n].sigmoid().max(1)
            #     labels[:, i][score_i <= cls_thresh] = -1
            #     pos += n

        #ids = score > cls_thresh
        keep = box_nms_fast(anchor_boxes, boxes, score, cls_thresh=cls_thresh, nms_thresh=nms_thresh)
        boxes, labels, score = boxes[keep].view(-1, 4), labels[keep].view(-1), score[keep].view(-1)

        keep = box_nms(boxes, score, threshold=nms_thresh)
        boxes, labels, score = boxes[keep], labels[keep], score[keep]

        return boxes, labels, score

def DataEncoderScripted():
    return torch.jit.script(DataEncoder)

def CreateDataEncoder(**kwargs):
    if (USE_FAST_DECODER and
            len(kwargs.get('anchor_areas', []))==1 and
            len(kwargs.get('aspect_ratios', []))==1 and
            len(kwargs.get('scale_ratios', []))==1):
        return DataEncoder(**kwargs)
    else:
        return pytorch_retinanet.encoder.DataEncoder(**kwargs)
