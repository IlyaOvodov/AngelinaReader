import sys
sys.path.append(r'..')
import local_config

from ovotools import AttrDict

param_file = 'NN_results/yolov3_828386.param.txt'
load_model_from = 'NN_results/yolov3_828386/models/02500.t7'

import os
param_file = os.path.join(local_config.data_path, param_file)
params = AttrDict.load(param_file)
params.data.batch_size = 7

params.model_params.img_size = 192
params.data.net_hw=(params.model_params.img_size, params.model_params.img_size)

params.load_model_from = load_model_from

device = 'cuda:0'

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools

import DSBI_invest.data
import my_models

from utils.utils import *

model, collate_fn, loss = my_models.create_model(params, 'test')
model = model.cuda()

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params, collate_fn = collate_fn)

val_loader1_it = iter(val_loader1)

data, target = next(val_loader1_it)

# boxes, scores  = model(data.cuda())
#r = my_models.do_nms(imgs = data, boxes = boxes, scores = scores, i = 0, max_per_image = 300)

conf_thres=0.2
nms_thres=0.2

model.eval()
with torch.no_grad():
    inf_out, train_out = model(data.cuda())
    output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

    for si, pred in enumerate(output):
        print(pred)