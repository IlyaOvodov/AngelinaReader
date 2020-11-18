#!/usr/bin/env python
# coding: utf-8
"""
Trains model using parameters and setting defined in model.params
Results are stored in local_config.data_path / params.model_name
"""
import local_config
import sys
sys.path.append(local_config.global_3rd_party)
from collections import OrderedDict
import os
import torch
import ignite
from ignite.engine import Events
from pathlib import Path

import ovotools.ignite_tools
import ovotools.pytorch_tools
import ovotools.pytorch

from data_utils import data
from model import create_model_retinanet
from model.params import params, settings
import model.validate_retinanet as validate_retinanet

if settings.findLR:
    params.model_name += '_findLR'
params.save(can_overwrite=settings.can_overwrite)


ctx = ovotools.pytorch.Context(settings=None, params=params)

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, device=settings.device)
if 'load_model_from' in params.keys():
    preloaded_weights = torch.load(Path(local_config.data_path) / params.load_model_from, map_location='cpu')
    model.load_state_dict(preloaded_weights)

ctx.net  = model
ctx.loss = loss

train_loader = data.create_dataloader(params, collate_fn,
                                            list_file_names=params.data.train_list_file_names, shuffle=True)
val_loaders = { k: data.create_dataloader(params, collate_fn, list_file_names=v, shuffle=False)
                for k,v in params.data.val_list_file_names.items() }
print('data loaded. train:{} batches'.format(len(train_loader)))
for k,v in val_loaders.items():
    print('             {}:{} batches'.format(k, len(v)))

ctx.optimizer = eval(params.optim)(model.parameters(), **params.optim_params)

metrics = OrderedDict({
    'loss': ignite.metrics.Loss(loss.metric('loss'), batch_size=lambda y: params.data.batch_size), # loss calc already called when train
    'loc': ignite.metrics.Loss(loss.metric('loc'), batch_size=lambda y: params.data.batch_size),
    'cls': ignite.metrics.Loss(loss.metric('cls'), batch_size=lambda y: params.data.batch_size),
})

eval_metrics = OrderedDict({
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: params.data.batch_size), # loss calc must be called when eval
    'loc': ignite.metrics.Loss(loss.metric('loc'), batch_size=lambda y: params.data.batch_size),
    'cls': ignite.metrics.Loss(loss.metric('cls'), batch_size=lambda y: params.data.batch_size),
})

target_metric = 'train:loss'

trainer_metrics = {} if settings.findLR else metrics
eval_loaders = {}
if settings.findLR:
    eval_loaders['train'] = train_loader
eval_loaders.update(val_loaders)
eval_event = ignite.engine.Events.ITERATION_COMPLETED if settings.findLR else ignite.engine.Events.EPOCH_COMPLETED
eval_duty_cycle = 2 if settings.findLR else 1
train_epochs = params.lr_finder.iters_num*len(train_loader) if settings.findLR else settings.max_epochs

trainer = ovotools.ignite_tools.create_supervised_trainer(model, ctx.optimizer, loss, metrics=trainer_metrics, device=settings.device)
evaluator = ignite.engine.create_supervised_evaluator(model, metrics=eval_metrics, device=settings.device)

if settings.findLR:
    best_model_buffer = None
else:
    best_model_buffer = ovotools.ignite_tools.BestModelBuffer(ctx.net, 'val:loss', minimize=True, params=ctx.params)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = eval_loaders,
                                                                best_model_buffer=best_model_buffer,
                                                                params = params,
                                                                duty_cycles = eval_duty_cycle)
trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

if settings.findLR:
    import math
    @trainer.on(Events.ITERATION_STARTED)
    def upd_lr(engine):
        log_lr = params.lr_finder.log_lr_start + (params.lr_finder.log_lr_end - params.lr_finder.log_lr_start) * (engine.state.iteration-1)/params.lr_finder.iters_num
        lr = math.pow(10, log_lr)
        ctx.optimizer.param_groups[0]['lr'] = lr
        engine.state.metrics['lr'] = ctx.optimizer.param_groups[0]['lr']
        if engine.state.iteration > params.lr_finder.iters_num:
            print('done')
            engine.terminate()
else:
    if params.lr_scheduler.type == 'clr':
        clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, ctx.optimizer, target_metric, params,
                                                       engine=trainer)
    else:
        ctx.create_lr_scheduler()

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler_step(engine):
            call_params = {'epoch': engine.state.epoch}
            if ctx.params.lr_scheduler.type.split('.')[-1] == 'ReduceLROnPlateau':
                call_params['metrics'] = engine.state.metrics['val_dsbi:loss']
            engine.state.metrics['lr'] = ctx.optimizer.param_groups[0]['lr']
            ctx.lr_scheduler.step(**call_params)

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_accuracy(engine):
        if engine.state.epoch % 100 == 1:
            data_set = validate_retinanet.prepare_data(ctx.params.data.val_list_file_names)
            for key, data_list in data_set.items():
                acc_res = validate_retinanet.evaluate_accuracy(os.path.join(ctx.params.get_base_filename(), 'param.txt'),
                                                               model, settings.device, data_list)
                for rk, rv in acc_res.items():
                    engine.state.metrics[key+ ':' + rk] = rv

#@trainer.on(Events.EPOCH_COMPLETED)
#def save_model(engine):
#    if save_every and (engine.state.epoch % save_every) == 0:
#        ovotools.pytorch_tools.save_model(model, params, rel_dir = 'models', filename = '{:05}.t7'.format(engine.state.epoch))

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})

tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters=settings.findLR)
tb_logger.start_server(settings.tensorboard_port, start_it = False)

@trainer.on(Events.ITERATION_COMPLETED)
def reset_resources(engine):
    engine.state.batch = None
    engine.state.output = None
    #torch.cuda.empty_cache()

trainer.run(train_loader, max_epochs = train_epochs)

