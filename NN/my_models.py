import sys
sys.path.append(r'..')
import local_config

sys.path.append(local_config.RFBNet)
import models.RFB_Net_mobile
import layers.modules

import os
import torch
import numpy as np
from utils.nms_wrapper import nms


num_classes = 64

def create_model(params, phase):
    model = models.RFB_Net_mobile.build_net(phase, params.data.net_hw[0], num_classes=num_classes).cuda()
    if 'load_model_from' in params.keys():
        preloaded_weights = torch.load(os.path.join(local_config.data_path, params.load_model_from))
        model.load_state_dict(preloaded_weights)

    criterion = layers.modules.MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

    priorbox = layers.functions.PriorBox(params.prior_box)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

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

    return model, loss


thresh = 0.1


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

