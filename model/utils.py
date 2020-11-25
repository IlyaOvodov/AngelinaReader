import torch

def meshgrid(x, y, row_major=True):
    # type: (int, int, bool)->Tensor
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1).float()
    yy = b.view(-1,1).repeat(1,x).view(-1,1).float()
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

def change_box_order(boxes, order):
    # type: (Tensor, str)->Tensor
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    xyxy2xywh: str = 'xyxy2xywh'
    xywh2xyxy: str = 'xywh2xyxy'
    assert order in [xyxy2xywh, xywh2xyxy]
    a = boxes[:,:,:2]
    b = boxes[:,:,2:]
    if order == xyxy2xywh:
        return torch.cat([(a+b)/2,b-a+1], 2)
    return torch.cat([a-b/2,a+b/2], 2)


def box_iou(box1, box2):
    # type: (Tensor, Tensor, str)->Tensor
    '''
    Compute the intersection over union of two set of boxes (element by element).
    Box order is (xmin, ymin, xmax, ymax [,...]).
    box1: (tensor) bounding boxes, sized [H,W,4+].
    box2: (tensor) bounding boxes, sized [H,W,4+].
    Return:
      (tensor) iou, sized [H,W].
    '''
    lt = torch.max(box1[:,:,:2], box2[:,:,:2])
    rb = torch.min(box1[:,:,2:4], box2[:,:,2:4])
    inter_wh = (rb-lt+1).clamp(min=0)
    inter = inter_wh[:,:,0] * inter_wh[:,:,1]
    area1 = (box1[:,:,2]-box1[:,:,0]+1) * (box1[:,:,3]-box1[:,:,1]+1)
    area2 = (box2[:,:,2]-box2[:,:,0]+1) * (box2[:,:,3]-box2[:,:,1]+1)
    iou = inter / (area1 + area2 - inter)
    return iou


def compare_neighbours(boxes1, scores1, alive_mask1, boxes2, scores2, alive_mask2, nms_thr):
    iou = box_iou(boxes1, boxes2)
    calc_area = iou > nms_thr
    first_is_stronger = calc_area & ((scores1 > scores2) | (scores1 == scores2) & (alive_mask1))
    second_is_stronger = calc_area & (scores1 < scores2)
    boxes2[first_is_stronger] = boxes1[first_is_stronger]
    scores2[first_is_stronger] = scores1[first_is_stronger]
    alive_mask2[first_is_stronger] = False
    boxes1[second_is_stronger] = boxes2[second_is_stronger]
    scores1[second_is_stronger] = scores2[second_is_stronger]
    alive_mask1[second_is_stronger] = False


def box_nms(anchor_boxes_wh, bboxes, scores, cls_thresh=0.5, nms_thresh=0.5):
    # type: (Tensor, Tensor, float, float)->Tensor
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      cls_thresh: (float) score threshold.
      nms_thresh: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.
    '''
    anchor_boxes = change_box_order(anchor_boxes_wh, 'xywh2xyxy')
    scores[bboxes[:,:,0]>anchor_boxes[:,:,2]] = 0
    scores[bboxes[:,:,1]>anchor_boxes[:,:,3]] = 0
    scores[bboxes[:,:,2]<anchor_boxes[:,:,0]] = 0
    scores[bboxes[:,:,3]<anchor_boxes[:,:,1]] = 0
    alive_mask = scores > cls_thresh
    prev_alive_cnt = 0
    alive_cnt = alive_mask.sum()
    side_neighbours = False
    rounds = 0
    while prev_alive_cnt != alive_cnt:
        prev_alive_cnt = alive_cnt
        if side_neighbours:
            compare_neighbours(bboxes[:-1,:], scores[:-1,:], alive_mask[:-1,:], bboxes[1:,:], scores[1:,:], alive_mask[1:,:], nms_thr=nms_thresh)
        else:
            compare_neighbours(bboxes[:,:-1], scores[:,:-1], alive_mask[:,:-1], bboxes[:,1:], scores[:,1:], alive_mask[:,1:], nms_thr=nms_thresh)
        side_neighbours = not side_neighbours
        alive_cnt = alive_mask.sum()
        rounds += 1
    return alive_mask.to(scores.device)

