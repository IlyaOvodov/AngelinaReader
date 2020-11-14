from typing import List
import torch
import pytorch_retinanet.encoder
from pytorch_retinanet.utils import box_nms

class PixelDataEncoder(pytorch_retinanet.encoder.DataEncoder):

    Scale = [10,5]

    def __init__(self, **kwargs):
        super(PixelDataEncoder, self).__init__(**kwargs)
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
        # debug
        #return super(PixelDataEncoder, self).decode(self, loc_preds, cls_preds, input_size, cls_thresh, nms_thresh, num_classes)

        assert loc_preds.device == cls_preds.device
        assert not isinstance(input_size, int)
        input_size = torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).to(loc_preds.device)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        if len(num_classes) <= 1:
            score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        else:
            score, _ = cls_preds.sigmoid().max(1)  # [#anchors,]
            labels = torch.zeros((len(cls_preds), len(num_classes)), dtype=torch.int, device=cls_preds.device)
            pos = 0
            for i, n in enumerate(num_classes):
                score_i, labels[:, i] = cls_preds[:, pos:pos+n].sigmoid().max(1)
                labels[:, i][score_i <= cls_thresh] = -1
                pos += n

        ids = score > cls_thresh
        ids = ids.nonzero().squeeze()             # [#obj,]
        if len(ids.shape):
            keep = self.box_nms(boxes[ids], score[ids], threshold=nms_thresh)
            return boxes[ids][keep], labels[ids][keep], score[ids][keep]
        else:
            return boxes[ids].unsqueeze(0), labels[ids].unsqueeze(0), score[ids].unsqueeze(0)

    def box_nms(self, bboxes, scores, threshold=0.5, mode='union'):
        # type: (Tensor, Tensor, float, str)->Tensor
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''

        if len(scores.shape) == 0:
            sz: List[int] = []
            return torch.tensor(sz, dtype=torch.long).to(scores.device)

        threshold = 2 * threshold  # NMS threshold -> occupied area threshold

        # bboxes = bboxes.to(dtype=torch.int16)
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1+1) * (y2-y1+1)

        board = torch.zeros((2000//self.Scale[0], 2000//self.Scale[1]), dtype=torch.bool)  # TODO size

        _, order = scores.sort(0, descending=True)

        keep: List[int] = []
        if len(order.shape) == 0:
            return torch.tensor(keep, dtype=torch.long).to(scores.device)
        for i in order:
            x1i = int(x1[i]) // self.Scale[1]
            y1i = int(y1[i]) // self.Scale[0]
            x2i = int(x2[i]) // self.Scale[1]
            y2i = int(y2[i]) // self.Scale[0]
            cross_area = board[y1i:y2i, x1i:x2i].any()
            if cross_area < areas[i] * threshold:
                 board[y1i:y2i, x1i:x2i] = 1
                 keep.append(i)

        return torch.tensor(keep, dtype=torch.long).to(scores.device)


