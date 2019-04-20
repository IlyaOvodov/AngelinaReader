from ovotools.params import AttrDict
import sys
sys.path.append('..')
sys.path.append('../NN/RetinaNet')
import local_config
from os.path import join
model_name = 'NN_results/retina_points_cde773'
model_fn = join(local_config.data_path, model_name)

params = AttrDict.load(model_fn + '.param.txt', verbose = True)
params.data.net_hw = (128,128,) #(512,768) ###### (1024,1536) #
params.data.batch_size = 1 #######

params.data.get_points = False
params.model_params.encoder_params.anchor_areas=[5 * 5., ] # 6 * 6., 10 * 10.,
params.model_params.encoder_params.iuo_fit_thr = 0 # if iou > iuo_fit_thr => rect fits anchor
params.model_params.encoder_params.iuo_nofit_thr = 0

params.augmentation = AttrDict(
    img_width_range=(1100, 1100),
    stretch_limit = 0,
    rotate_limit=0,
)

import ignite

device = 'cuda:0'


# In[3]:

import torch
import DSBI_invest.data
import create_model_retinanet

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)
model = None

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params, collate_fn,
                            data_dir=r'D:\Programming\Braille\Data\My', fn_suffix = '', mode='inference',
                                                                               #mode = 'debug',
                                                                               verbose = 2)
val_loader1_it = iter(train_loader)

import PIL
import PIL.ImageDraw
import numpy as np
def TensorToPilImage(tensor, params):
    std = (0.2,) #params.data.std
    mean = (0.5,) #params.data.mean
    vx_np = tensor.cpu().numpy().copy()
    vx_np *= np.asarray(std)[:, np.newaxis, np.newaxis]
    vx_np += np.asarray(mean)[:, np.newaxis, np.newaxis]
    vx_np = vx_np.transpose(1,2,0)*255
    return PIL.Image.fromarray(vx_np.astype(np.uint8))

encoder = loss.encoder

batch = next(val_loader1_it)
data, target = ignite.engine._prepare_batch(batch, device=device)

cls_thresh = 0.6
nms_thresh = 0

img = TensorToPilImage(data[0], params)
w,h = img.size

labels = target[1][0].clamp(min=0)
ly = torch.eye(65, device = labels.device)  # [D,D]
cls_preds = ly[labels][:,1:]

boxest, labelst, scores = encoder.decode(target[0][0].cpu().data, cls_preds, (w,h),
                              cls_thresh = cls_thresh, nms_thresh = nms_thresh)
print(len(boxest))
