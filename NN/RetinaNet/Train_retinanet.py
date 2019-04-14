import sys
sys.path.append(r'../..')
import local_config

from ovotools import AttrDict

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/retina_chars',
    data = AttrDict(
        get_points = False,
        batch_size = 10,
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        net_hw = (416, 416)
    ),
    model = 'retina',
    model_params = AttrDict(
        encoder_params = AttrDict(
            anchor_areas = [8*8., 16*16., 32*32., 64*64., 128*128.],
        ),
    ),
    #load_model_from = 'NN_results/RFBNet0_db0807.model/00500.t7',
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=0.0001,
        #momentum=0,
        #weight_decay = 0, #0.001,
        #nesterov = False,
    ),
    lr_finder=AttrDict(
        epochs_num=100,
        log_lr_start=-6,
        log_lr_end=-1,
    ),
    clr=AttrDict(
        warmup_epochs=10,
        min_lr=4e-6,
        max_lr=7e-4,
        period_epochs=200,
        # scale_max_lr=0.95,
        # scale_min_lr=1,
    ),

    decimate_lr_every = 1200,
)
max_epochs = 10000
tensorboard_port = 6006
save_every = 500
device = 'cuda:0'
findLR = False

if findLR:
    params.model_name += '_findLR'

params.save(can_overwrite = True)

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools

import DSBI_invest.data
import create_model_retinanet

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params, collate_fn)

optimizer = eval(params.optim)(model.parameters(), **params.optim_params)

metrics = {
    'loss': ignite.metrics.Loss(loss.metric('loss'), batch_size=lambda y: params.data.batch_size),
    'loc': ignite.metrics.Loss(loss.metric('loc'), batch_size=lambda y: params.data.batch_size),
    'cls': ignite.metrics.Loss(loss.metric('cls'), batch_size=lambda y: params.data.batch_size),
}

target_metric = 'val:loss'

trainer_metrics = {} if findLR else metrics
eval_loaders = {'val': val_loader1, 'val2': val_loader2}
if findLR:
    eval_loaders['train'] = train_loader
eval_event = ignite.engine.Events.ITERATION_COMPLETED if findLR else ignite.engine.Events.EPOCH_COMPLETED
eval_duty_cycle = 2 if findLR else 5
train_epochs = params.lr_finder.epochs_num if findLR else max_epochs

trainer = ovotools.ignite_tools.create_supervised_trainer(model, optimizer, loss, metrics=trainer_metrics, device = device)
evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device = device)

log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                #loaders_dict = {},
                                                                loaders_dict = eval_loaders,
                                                                best_model_buffer=None,
                                                                params = params,
                                                                duty_cycles = eval_duty_cycle)
trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

if findLR:
    import math
    lr_find_len = len(train_loader)*params.lr_finder.epochs_num
    @trainer.on(Events.ITERATION_STARTED)
    def upd_lr(engine):
        log_lr = params.lr_finder.log_lr_start + (params.lr_finder.log_lr_end - params.lr_finder.log_lr_start) * (engine.state.iteration-1)/lr_find_len
        lr = math.pow(10, log_lr)
        optimizer.param_groups[0]['lr'] = lr
        engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
else:
    clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, optimizer, target_metric, params,
                                                       engine=trainer)
    #@trainer.on(Events.EPOCH_COMPLETED)
    #def save_model(engine):
    #    if save_every and (engine.state.epoch % save_every) == 0:
    #        ovotools.pytorch_tools.save_model(model, params, rel_dir = 'models', filename = '{:05}.t7'.format(engine.state.epoch))

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})

tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters = findLR)
tb_logger.start_server(tensorboard_port, start_it = False)

trainer.run(train_loader, max_epochs = train_epochs)

