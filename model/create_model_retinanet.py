import os
import torch
import numpy as np

from pytorch_retinanet.loss import FocalLoss
from pytorch_retinanet.retinanet import RetinaNet
from pytorch_retinanet.encoder import DataEncoder

from braille_utils import label_tools

def create_model_retinanet(params, device):
    '''
    Creates model and auxiliary functions
    :param params: OvoTools.AttrDict with parameters
    :param device: 'cuda'/'cpu'
    :return: model, detection_collate function, loss function
    '''
    use_multiple_class_groups = params.data.get('class_as_6pt', False)
    num_classes = 1 if params.data.get_points else ([1]*6 if use_multiple_class_groups else 64)
    encoder = DataEncoder(**params.model_params.encoder_params)
    two_heads = params.data.get('load_front_side', False) and params.data.get('load_front_side', False)
    model = RetinaNet(num_layers=encoder.num_layers(), num_anchors=encoder.num_anchors(),
                      num_classes=num_classes,
                      num_fpn_layers=params.model_params.get('num_fpn_layers', 0),
                      fpn_skip_layers=params.model_params.encoder_params.get('fpn_skip_layers', 0),
                      num_heads=2 if two_heads else 1
                      ).to(device)
    retina_loss = FocalLoss(num_classes=num_classes, **params.model_params.get('loss_params', dict()))

    def detection_collate(batch):
        '''
        :param batch: list of (tb image(CHW float), [(left, top, right, bottom, class),...]) сcoords in [0,1], extra_params
        :return: batch: ( images (BCNHW), ( encoded_rects, encoded_labels ) )
        copied from RetinaNet, but a) accepts rects as input, b) returns (x,y) where y = (encoded_rects, encoded_labels)
        '''

        # t = [b for b in batch if b[1].shape[0]==0]
        # if len(t):
        #     pass

        #device = torch.device('cpu')  # commented to use settings.device

        boxes = [torch.tensor(b[1][:, :4], dtype = torch.float32, device=device)
                 *torch.tensor(params.data.net_hw[::-1]*2, dtype = torch.float32, device=device) for b in batch]
        labels = [torch.tensor(b[1][:, 4], dtype = torch.long, device=device) for b in batch]
        scores = [torch.tensor(b[1][:, 5], dtype = torch.float32, device=device) for b in batch]
        if params.data.get_points:
            labels = [torch.tensor([0]*len(lb), dtype = torch.long, device=device) for lb in labels]
        elif use_multiple_class_groups:
            # классы нумеруются с 0, отсутствие класса = -1, далее в encode cls_targets=1+labels
            labels = [torch.tensor([[int(ch)-1 for ch in label_tools.int_to_label010(int_lbl.item())] for int_lbl in lb],
                                   dtype=torch.long, device=device) for lb in labels]

        original_images = [b[3] for b in batch if len(b)>3] # batch contains augmented image if not in train mode

        imgs = [x[0] for x in batch]
        calc_cls_mask = torch.tensor([b[2].get('calc_cls', True) for b in batch],
                                dtype=torch.bool,
                                device=device)
        is_front_side_mask = torch.tensor([b[2]['is_front_side'] for b in batch],
                                dtype=torch.bool,
                                device=device)

        h, w = tuple(params.data.net_hw)
        num_imgs = len(batch)
        inputs = torch.zeros(num_imgs, 3, h, w).to(imgs[0])

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            labels_i = labels[i]
            if use_multiple_class_groups and len(labels_i.shape) != 2:  # it can happen if no labels are on image
                labels_i = labels_i.reshape((0, len(num_classes)))
            loc_target, cls_target, max_ious = encoder.encode(boxes[i], labels_i, input_size=(w,h), scores=scores[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if original_images: # inference mode
            return inputs, ( torch.stack(loc_targets), torch.stack(cls_targets), calc_cls_mask, is_front_side_mask), original_images
        else:
            return inputs, (torch.stack(loc_targets), torch.stack(cls_targets), calc_cls_mask, is_front_side_mask)

    class Loss:
        def __init__(self):
            self.encoder = encoder
            self.loss_module = retina_loss
            pass
        def __call__(self, pred, targets):
            if isinstance(pred[0], torch.Tensor):  # num_heads==1
                loc_preds, cls_preds = pred
                loc_targets, cls_targets, calc_cls_mask, is_front_side_mask = targets
                if calc_cls_mask.min():  # Ничего не пропускаем
                    calc_cls_mask = None
                loss = retina_loss(loc_preds, loc_targets, cls_preds, cls_targets, cls_calc_mask=calc_cls_mask)
            else:
                loss = 0
                assert len(pred) == 2
                for pred_i, handle_front_side in zip(pred, (True, False)):
                    loc_preds, cls_preds = pred_i
                    loc_targets, cls_targets, calc_cls_mask, is_front_side_mask = targets
                    this_side_mask = is_front_side_mask == handle_front_side
                    calc_cls_mask_i = calc_cls_mask & this_side_mask
                    if this_side_mask.max() or calc_cls_mask_i.max():
                        if calc_cls_mask_i.min():  # Ничего не пропускаем
                            calc_cls_mask_i = None
                        if this_side_mask.min():
                            this_side_mask = None
                        loss += retina_loss(loc_preds, loc_targets, cls_preds, cls_targets,
                                        loc_calc_mask=this_side_mask, cls_calc_mask=calc_cls_mask_i)
            return loss

        def get_dict(self, *kargs, **kwargs):
            return retina_loss.loss_dict
        def metric(self, key):
            def call(*kargs, **kwargs):
                return retina_loss.loss_dict.get(key, torch.zeros(1)[0])
            return call

    return model, detection_collate, Loss()

if __name__ == '__main__':
    pass
