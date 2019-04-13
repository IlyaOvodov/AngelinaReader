#!/usr/bin/env python
# coding: utf-8

# # Ïðîâåðêà ðàáîòû íåéðîñåòè äëÿ îáëàñòåé îäèíàêîâûõ òîâàðîâ
# ðèñîâàíèå êàðòèíîê
# âû÷èñëåíèå symmetric_best_dice

# In[1]:


from ovotools.params import AttrDict
import sys
sys.path.append('../..')
import local_config
from os.path import join
model_name = 'NN_results/retina_chars_b24228'
model_fn = join(local_config.data_path, model_name)

params = AttrDict.load(model_fn + '.param.txt', verbose = True)
model_fn += '/models/02500.t7'
#params.data.net_hw = (416, 416) #(512,768) ###### (1024,1536) #
params.data.batch_size = 1 #######

import torch
import ignite

device = 'cuda:0'


# In[3]:

import DSBI_invest.data
import create_model_retinanet

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)
model = model.to(device)
model.load_state_dict(torch.load(model_fn))
model.eval()
print("Model loaded")

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params, collate_fn)
val_loader1_it = iter(val_loader1)
batch = next(val_loader1_it)
data, target = ignite.engine._prepare_batch(batch, device=device)

with torch.no_grad():
    (loc_preds, cls_preds) = model(data.to(device))

loss_val = loss((loc_preds, cls_preds), target)
loss_val, loss.get_dict()

import PIL
import PIL.ImageDraw

import numpy as np
def TensorToPilImage(tensor, params):
    vx_np = tensor.cpu().numpy().copy()
    vx_np *= np.asarray(params.data.std)[:, np.newaxis, np.newaxis]
    vx_np += np.asarray(params.data.mean)[:, np.newaxis, np.newaxis]
    vx_np = vx_np.transpose(1,2,0)*255
    return PIL.Image.fromarray(vx_np.astype(np.uint8))

img = TensorToPilImage(data[0], params)
w,h = img.size
encoder = loss.encoder
boxes, labels = encoder.decode(loc_preds[0].cpu().data, cls_preds[0].cpu().data, (w,h))

draw = PIL.ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img

pass

