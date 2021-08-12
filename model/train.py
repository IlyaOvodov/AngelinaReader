#!/usr/bin/env python
# coding: utf-8
"""
Trains model using parameters and setting defined in model.params
Results are stored in local_config.data_path / params.model_name
"""
import local_config
import sys
from collections import OrderedDict
import os
import torch
import ignite
from ignite.engine import Events
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import ovotools.ignite_tools
import ovotools.pytorch_tools
import ovotools.pytorch

from data_utils import data
from model import create_model_retinanet
from model.params import params, settings
import model.validate_retinanet as validate_retinanet

if len(sys.argv) > 2:
    dir = Path(sys.argv[1])
    if not dir.is_absolute():
        dir = Path(local_config.data_path) / dir
    checkpoint = str(dir/'checkpoint') + '_{}_'+sys.argv[2]+'.pth'
else:
    checkpoint =  None

def load_objects (to_load, checkpoint):
    for k, obj in to_load.items():
        f_name = checkpoint.format(k)
        state_dict = torch.load(f_name)
        obj.load_state_dict(state_dict)

if checkpoint:
    load_objects(to_load={"settings": settings, "params": params}, checkpoint=checkpoint)
else:
    if settings.findLR:
        params.model_name += '_findLR'
    params.save(can_overwrite=settings.can_overwrite)

ctx = ovotools.pytorch.Context(settings=None, params=params, eval_func=lambda x: eval(x))

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, device=settings.device)
if checkpoint:
    load_objects(to_load={"model": model}, checkpoint=checkpoint)
else:
    if 'load_model_from' in params.keys():
        preloaded_weights = torch.load(Path(local_config.data_path) / params.load_model_from, map_location='cpu')

        keys_to_replace = []
        for k in preloaded_weights.keys():
            if k[:8] in ('loc_head', 'cls_head'):
                keys_to_replace.append(k)
        for k in keys_to_replace:
            k2 = k.split('.')
            if len(k2) == 4:
                k2 = k2[0] + '.' + k2[2] + '.' + k2[3]
                preloaded_weights[k2] = preloaded_weights.pop(k)

        model.load_state_dict(preloaded_weights)

ctx.net  = model
ctx.loss = loss

train_loader = data.create_dataloader(params, device=settings.device, collate_fn=collate_fn,
                                            list_file_names=params.data.train_list_file_names, shuffle=True)
val_loaders = { k: data.create_dataloader(params, device=settings.device, collate_fn=collate_fn, list_file_names=v, shuffle=False)
                for k,v in params.data.val_list_file_names.items() }
print('data loaded. train:{} batches'.format(len(train_loader)))
for k,v in val_loaders.items():
    print('             {}:{} batches'.format(k, len(v)))

ctx.optimizer = eval(params.optim)(model.parameters(), **params.optim_params)
if checkpoint:
    load_objects(to_load={"optimizer": ctx.optimizer}, checkpoint=checkpoint)

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

def save_model(model, rel_dir="models", filename="last.t7"):
    file_name = os.path.join(ctx.params.get_base_filename(), rel_dir, filename)
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), file_name)

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
    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_accuracy(engine):
        if engine.state.epoch % 100 == 1:
            data_set = validate_retinanet.prepare_data(ctx.params.data.val_list_file_names)
            for key, data_list in data_set.items():
                acc_res = validate_retinanet.evaluate_accuracy(os.path.join(ctx.params.get_base_filename(), 'param.txt'),
                                                               model, settings.device, data_list)
                for rk, rv in acc_res.items():
                    engine.state.metrics[key+ ':' + rk] = rv

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model_on_event(engine):
        if not settings.get("regular_save_period"):
            return
        try:
            period, step_in_epoch = settings.regular_save_period
        except:  # not tuple
            period, step_in_epoch = settings.regular_save_period, 1
        if engine.state.epoch % period == step_in_epoch and engine.state.epoch > period:
            save_model(model, filename="{:06}.t7".format(engine.state.epoch))


    clr_scheduler = None
    if params.lr_scheduler.type == 'clr':
        clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, ctx.optimizer, target_metric, params,
                                                       engine=trainer)
    else:
        ctx.create_lr_scheduler()

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler_step(engine):
            call_params = {'epoch': engine.state.epoch}
            if ctx.params.lr_scheduler.type.split('.')[-1] == 'ReduceLROnPlateau':
                call_params['metrics'] = engine.state.metrics[ctx.params.data.target_metric]
            engine.state.metrics['lr'] = ctx.optimizer.param_groups[0]['lr']
            ctx.lr_scheduler.step(**call_params)

    if checkpoint:
        load_objects(to_load={"lr_scheduler": clr_scheduler or ctx.lr_scheduler}, checkpoint=checkpoint)

if settings.findLR:
    best_model_buffer = None
else:
    best_model_buffer = ovotools.ignite_tools.BestModelBuffer(ctx.net, ctx.params.data.target_metric, minimize=False, params=ctx.params)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = eval_loaders,
                                                                best_model_buffer=best_model_buffer,
                                                                params = params,
                                                                duty_cycles = eval_duty_cycle)
trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

checkpoint_objects = {"settings": settings,
                      'params': params,
                      'model': model,
                      'optimizer': ctx.optimizer,
                      'lr_scheduler': clr_scheduler or ctx.lr_scheduler,
                      'trainer': trainer}
if best_model_buffer:
    checkpoint_objects.update({"best_model_buffer": best_model_buffer})
if checkpoint:
    load_objects(to_load=checkpoint_objects, checkpoint=checkpoint)

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

trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          ignite.handlers.ModelCheckpoint(
                            dirname=ctx.params.get_base_filename(),
                            filename_prefix='checkpoint',
                            save_interval=10, n_saved=1,
                            atomic=True, require_empty=False,
                            create_dir=False,
                            save_as_state_dict=True)
                          , checkpoint_objects)

trainer.run(train_loader, max_epochs = train_epochs)

save_model(model, filename="last.t7")