import sys
sys.path.append(r'..')
import local_config

import os
import torch
import numpy as np


#num_classes = 64

def create_model(params, phase):
    '''
    :param params:
    :param phase: 'train'/'test'
    :return: model, loss
    '''
    model_name = params.get('model', 'RFB_Net')
    if model_name =='RFB_Net':
        return create_model_RFB_Net(params, phase)
    elif model_name =='yolov3':
        return create_model_yolov3(params, phase)
    else:
        assert False, model_name



def create_model_RFB_Net(params, phase):
    sys.path.append(local_config.global_3rd_party + '/RFBNet')
    import models.RFB_Net_mobile
    import layers.modules

    model = models.RFB_Net_mobile.build_net(phase, params.data.net_hw[0], num_classes=num_classes).cuda()
    if 'load_model_from' in params.keys():
        preloaded_weights = torch.load(os.path.join(local_config.data_path, params.load_model_from))
        model.load_state_dict(preloaded_weights)

    criterion = layers.modules.MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

    priorbox = layers.functions.PriorBox(params.prior_box)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    def detection_collate(batch):
        '''
        :param batch: list of (tb image(CHW float), [(left, top, right, bottom, class),...])
        :return: batch: ( images (BCNHW), [rects1: numpy Nx5, ... rectsB] )
        '''
        targets = []
        imgs = []
        for _, (img, rects) in enumerate(batch):
            assert torch.is_tensor(img)
            imgs.append(img)
            assert isinstance(rects, type(np.empty(0)))
            annos = torch.from_numpy(rects).float()
            targets.append(annos)
        return (torch.stack(imgs, 0), targets)

    class Loss:
        def __init__(self):
            pass
        def __call__(self, pred, targets):
            targets = [anno.cuda() for anno in targets]
            self.loss_l, self.loss_c = criterion(pred, priors, targets)
            self.loss = self.loss_l + self.loss_c
            return self.loss
        def get_loss_l(self, *kargs, **kwargs):
            return self.loss_l
        def get_loss_c(self, *kargs, **kwargs):
            return self.loss_c

    loss = Loss()

    return model, detection_collate, loss


def create_model_yolov3(params, phase):
    sys.path.append(local_config.global_3rd_party + '/yolov3')
    from models import Darknet
    from utils.utils import model_info
    from utils.utils import build_targets
    from utils.utils import compute_loss

    model = Darknet(cfg_path = params.model_params.cfg, img_size = params.data.net_hw[0]).cuda()
    #model_info(model)

    def detection_collate(batch):
        '''
        :param batch: list of (tb image(CHW float), [(left, top, right, bottom, class),...])
        :return: batch: ( images (BCNHW), rects1: img_i, class, cx, cy, w, h )
        '''
        targets = []
        imgs = []
        for bi, (img, rects) in enumerate(batch):
            assert torch.is_tensor(img)
            imgs.append(img)
            assert isinstance(rects, type(np.empty(0)))

            if rects.shape != (0,):
                annos = np.empty((rects.shape[0], 6), dtype=np.float32)
                annos[:, 0:1] = bi
                annos[:, 1:2] = rects[:, 4:5]
                annos[:, 2:3] = (rects[:, 0:1] + rects[:, 2:3])/2
                annos[:, 3:4] = (rects[:, 1:2] + rects[:, 3:4])/2
                annos[:, 4:5] = (rects[:, 2:3] - rects[:, 0:1])
                annos[:, 5:6] = (rects[:, 3:4] - rects[:, 1:2])
                targets.append(torch.from_numpy(annos))
            if len(targets):
                targets_out = torch.cat(targets, 0)
            else:
                targets_out = torch.empty((0,6), dtype = torch.float32)
        return (torch.stack(imgs, 0), targets_out)

    class Loss:
        def __init__(self):
            self.loss_dict = dict()
            pass
        def __call__(self, pred, targets):
            target_list = build_targets(model, targets)
            loss, self.loss_dict = compute_loss(pred, target_list)
            return loss.squeeze()
        def get_dict(self, *kargs, **kwargs):
            return self.loss_dict
        def metric(self, key):
            def call(*kargs, **kwargs):
                return torch.tensor(self.loss_dict[key])
            return call


    loss = Loss()

    return model, detection_collate, loss



thresh = 0.1

'''
sys.path.append(local_config.global_3rd_party + '/RFBNet')
from utils.nms_wrapper import nms

def do_nms(imgs, boxes, scores, i, max_per_image):
    all_boxes = [[[] for _ in range(imgs.shape[0])]
                 for _ in range(num_classes)]

    img = imgs[i]
    boxes = boxes[i].detach()
    scores_per_image = boxes.shape[0]
    scores = scores[scores_per_image*i : scores_per_image*(i+1)].detach() #TODO

    scale = torch.Tensor([img.shape[-1], img.shape[-2],
                          img.shape[-1], img.shape[-2]]).cuda()

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    # scale each detection back up to the image

    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(c_dets, 0.45, force_cpu=True)
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

'''