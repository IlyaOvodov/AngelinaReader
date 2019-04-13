import os
import torch
import numpy as np

import sys
sys.path.append(r'..')
import local_config

def create_model_yolov3(params, phase, device):
    sys.path.append(local_config.global_3rd_party + '/yolov3')
    from models import Darknet
    from utils.utils import model_info
    from utils.utils import build_targets
    from utils.utils import compute_loss

    model = Darknet(cfg_path = params.model_params.cfg, img_size = params.data.nn_image_size_hw[0]).to(device)
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
                annos[:, 0] = bi
                annos[:, 1] = 0 # ignore class # rects[:, 4]
                annos[:, 2] = (rects[:, 0] + rects[:, 2])/2
                annos[:, 3] = (rects[:, 1] + rects[:, 3])/2
                annos[:, 4] = (rects[:, 2] - rects[:, 0])
                annos[:, 5] = (rects[:, 3] - rects[:, 1])
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
            if not model.training:
                pred = pred[1]
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

if __name__ == '__main__':
    pass
