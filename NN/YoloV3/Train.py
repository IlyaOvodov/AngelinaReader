#!/usr/bin/env python
# coding: utf-8
import imp
import local_config; local_config = imp.reload(local_config)
import ovotools.params; ovotools.params = imp.reload(ovotools.params)
import ovotools.pytorch_tools; ovotools.pytorch_tools = imp.reload(ovotools.pytorch_tools)
import ovotools.ignite_tools;  ovotools.ignite_tools = imp.reload(ovotools.ignite_tools)
from ovotools.params import AttrDict

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'yolov3/instances',
    data = AttrDict(
        fold_test_substrs = ['/cam_7_7/', '/cam_7_8/', '/cam_7_9/'],
        fold_no = 0,

        downscale_ratio = 1,
        batch_size = 10,
        #batch_size_test = 4,
        shuffle = True,
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        nn_image_size_hw = (416, 416),
    ),
    augm_params = AttrDict(
        random_scale = 0.3,
        random_rotate = 10,
    ),
    model = 'yolov3',
    model_params = AttrDict( {
        'cfg': 'yolov3.cfg',
        'img_size': 416
    } ),
    prior_box=AttrDict({
        'feature_maps': [19, 10, 5, 3, 2, 1],
        'min_dim': 300,
        'steps': [16, 32, 64, 100, 150, 300],
        'min_sizes': [45, 90, 135, 180, 225, 270],
        'max_sizes': [90, 135, 180, 225, 270, 315],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
    }),
    #load_weights_from = 'model/inner_halo',

    optimizer = 'torch.optim.Adam',
    optimizer_params = AttrDict(
        lr=0.001,
        #momentum=0,
        #weight_decay = 0.001,
        #nesterov = False,
    ),
    decimate_lr_every = 1200, # original 3000
    lr_finder = AttrDict(
        epochs_num = 6,
        log_lr_start = -3.5,
        log_lr_end = 0,
    ),
    clr=AttrDict(
        warmup_epochs=10,
        min_lr=0.0023,
        max_lr=0.06,
        period_epochs=50,
        #scale_max_lr=0.95,
        #scale_min_lr=1,
    ),

    pin_memory = False,
    demo_data_len = None,
)

gpus = [2] #[2]
max_epochs = 100000
tensorboard_port = 6006
save_every = 100
DemoMode = True

#if DemoMode:
#    params.demo_data_len = 10
#    params.data.batch_size = 12
#    params.data.samples_per_class = 3

if gpus is None:
    params['gpus_count'] = 1
else:
    params['gpus_count'] = len(gpus)


# In[4]:


import os
import sys
import random
import copy
import numpy as np
import PIL
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as F
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
if gpus is not None:
    assert type(gpus) and len(gpus) >= 1 and torch.cuda.is_available(), (type(gpus), len(gpus), torch.cuda.is_available())
    device = device+':'+str(gpus[0])


# In[5]:


from imp import reload
import data; data = reload(data)
import inner_halo; inner_halo = reload(inner_halo)
import create_model_yolov3; create_model_yolov3 = reload(create_model_yolov3)

import local_config; local_config = reload(local_config)


# In[7]:

params.save(can_overwrite = DemoMode, verbose = 1)

# In[10]:


model, detection_collate, loss = create_model_yolov3.create_model_yolov3(params, 'train', device=device)
model = model.to(device)
if gpus and len(gpus) > 1:
    model = torch.nn.DataParallel(model)

train_loader, val_loader = data.CreateAlbuLoadersV3(params, workers = 0, collate_fn=detection_collate, bboxes = True)


optimizer = eval(params.optimizer)(model.parameters(), **params.optimizer_params)

metrics = {
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: params.data.batch_size),
    'xy': ignite.metrics.Loss(loss.metric('xy'), batch_size=lambda y: params.data.batch_size),
    'wh': ignite.metrics.Loss(loss.metric('wh'), batch_size=lambda y: params.data.batch_size),
    'conf': ignite.metrics.Loss(loss.metric('conf'), batch_size=lambda y: params.data.batch_size),
    'cls': ignite.metrics.Loss(loss.metric('cls'), batch_size=lambda y: params.data.batch_size),
}

target_metric = 'val:loss'

trainer = ovotools.ignite_tools.create_supervised_trainer(model, optimizer, loss, metrics=metrics, device = device)
evaluator = create_supervised_evaluator(model, metrics=metrics, device = device)

log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = {'val': val_loader},
                                                                best_model_buffer=None,
                                                                params = params)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, event = Events.EPOCH_COMPLETED)



#clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, optimizer, target_metric, params, engine = trainer)

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def upd_lr(engine):
    if engine.state.epoch % params.decimate_lr_every == 0:
        optimizer.param_groups[0]['lr'] /= 10
    engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']

@trainer.on(Events.EPOCH_COMPLETED)
def save_model(engine):
    if save_every and (engine.state.epoch % save_every) == 0:
        ovotools.pytorch_tools.save_model(model, params, rel_dir = 'models', filename = '{:05}.t7'.format(engine.state.epoch))

tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters = False)
tb_logger.start_server(tensorboard_port)

trainer.run(train_loader, max_epochs = max_epochs)

