import sys
sys.path.append(r'..')
import local_config

from ovotools import AttrDict

param_file = 'NN_results/RFBNet0_db0807.param.txt'
load_model_from = 'NN_results/RFBNet0_db0807.model/00500.t7'

import os
param_file = os.path.join(local_config.data_path, param_file)
params = AttrDict.load(param_file)
params.load_model_from = load_model_from

device = 'cuda:0'

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools

import DSBI_invest.data
import my_models

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params)

model, loss = my_models.create_model(params, 'test')

val_loader1_it = iter(val_loader1)

data, target = next(val_loader1_it)

boxes, scores  = model(data.cuda())

pass



r = my_models.do_nms(imgs = data, boxes = boxes, scores = scores, i = 0, max_per_image = 300)
pass