import os
import torch
import numpy as np

import sys
sys.path.append(r'..')
import local_config

def create_model_retinanet(params, phase, device):
    sys.path.append(local_config.global_3rd_party)
    from pytorch_retinanet.loss import FocalLoss
    from pytorch_retinanet.retinanet import RetinaNet
    from pytorch_retinanet.encoder import DataEncoder

    num_classes = 1 if params.data.get_points else 64
    encoder = DataEncoder(**params.model_params.encoder_params)
    model = RetinaNet(num_layers=encoder.num_layers(), num_anchors=encoder.num_anchors(),
                      num_classes=num_classes).to(device)
    retina_loss = FocalLoss(num_classes=num_classes, **params.model_params.get('loss_params', dict()))

    if 'load_model_from' in params.keys():
        preloaded_weights = torch.load(os.path.join(local_config.data_path, params.load_model_from))
        model.load_state_dict(preloaded_weights)


    def detection_collate(batch):
        '''
        :param batch: list of (tb image(CHW float), [(left, top, right, bottom, class),...]) сcoords in [0,1]
        :return: batch: ( images (BCNHW), ( encoded_rects, encoded_labels ) )
        copied from RetinaNet, but a) accepts rects as input, b) returns (x,y) where y = (encoded_rects, encoded_labels)
        '''

        t = [b for b in batch if b[1].shape[0]==0]
        if len(t):
            pass

        boxes = [torch.tensor(b[1][:, :4], dtype = torch.float32)
                 *torch.tensor(params.data.net_hw[::-1]*2, dtype = torch.float32) for b in batch]
        labels = [torch.tensor(b[1][:, 4], dtype = torch.long) for b in batch]
        if params.data.get_points:
            labels = [torch.tensor([0]*len(lb), dtype = torch.long) for lb in labels]
        original_images = [b[2] for b in batch if len(b)>2] # batch contains augmented image if not in train mode

        imgs = [x[0] for x in batch]

        h, w = tuple(params.data.net_hw)
        num_imgs = len(batch)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target, max_ious = encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if original_images: # inference mode
            return inputs, ( torch.stack(loc_targets), torch.stack(cls_targets) ), original_images
        else:
            return inputs, (torch.stack(loc_targets), torch.stack(cls_targets))

    class Loss:
        def __init__(self):
            self.encoder = encoder
            pass
        def __call__(self, pred, targets):
            loc_preds, cls_preds = pred
            loc_targets, cls_targets = targets
            loss = retina_loss(loc_preds, loc_targets, cls_preds, cls_targets)
            return loss
        def get_dict(self, *kargs, **kwargs):
            return retina_loss.loss_dict
        def metric(self, key):
            def call(*kargs, **kwargs):
                return retina_loss.loss_dict[key]
            return call

    return model, detection_collate, Loss()

if __name__ == '__main__':
    pass
