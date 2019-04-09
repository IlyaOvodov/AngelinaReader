import sys
sys.path.append(r'..')
import local_config

from ovotools import AttrDict

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/yolov3_points',
    data = AttrDict(
        get_points = True,
        batch_size = 10,
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        net_hw = (256,256)
    ),
    model = 'yolov3',
    model_params = AttrDict( {
        'cfg': 'yolov3_points1.cfg',
    }),
    # for RFB net
    #prior_box = AttrDict( {
    #    'feature_maps': [19, 10, 5, 3, 2, 1],
    #    'min_dim': 300,
    #    'steps': [16, 32, 64, 100, 150, 300],
    #    'min_sizes': [45, 90, 135, 180, 225, 270],
    #    'max_sizes': [90, 135, 180, 225, 270, 315],
    #    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    #    'variance': [0.1, 0.2],
    #    'clip': True,
    #}),
    #load_model_from = 'NN_results/RFBNet0_db0807.model/00500.t7',
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=1e-3,# momentum=0.9,
        #weight_decay=1e-3
    ),
)
max_epochs = 100000
tensorboard_port = 6006
save_every = 500
device = 'cuda:0'

params.save(can_overwrite = True)

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools

import DSBI_invest.data
import my_models

model, collate_fn, loss = my_models.create_model(params, 'train')

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params, collate_fn)

optimizer = eval(params.optim)(model.parameters(), **params.optim_params)

metrics = {
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: params.data.batch_size),
    'xy': ignite.metrics.Loss(loss.metric('xy'), batch_size=lambda y: params.data.batch_size),
    'wh': ignite.metrics.Loss(loss.metric('wh'), batch_size=lambda y: params.data.batch_size),
    'conf': ignite.metrics.Loss(loss.metric('conf'), batch_size=lambda y: params.data.batch_size),
    'cls': ignite.metrics.Loss(loss.metric('cls'), batch_size=lambda y: params.data.batch_size),
}

trainer = ovotools.ignite_tools.create_supervised_trainer(model, optimizer, loss, metrics=metrics, device = device)
evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device = device)

best_model_buffer = ovotools.ignite_tools.BestModelBuffer(model, 'train:loss', params, verbose = 0)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                #loaders_dict = {},
                                                                loaders_dict = {'val': val_loader1, 'val2': val_loader2},
                                                                best_model_buffer=None, #best_model_buffer,
                                                                params = params,
                                                                duty_cycles = 5)
trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, log_training_results, event = ignite.engine.Events.EPOCH_COMPLETED)

#clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, optimizer, 'train:loss', params, engine = trainer)

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def upd_lr(engine):
    if engine.state.epoch % 3000 == 0:
        optimizer.param_groups[0]['lr'] /= 10
        engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']


@trainer.on(Events.EPOCH_COMPLETED)
def save_model(engine):
    if save_every and (engine.state.epoch % save_every) == 0:
        ovotools.pytorch_tools.save_model(model, params, rel_dir = 'models', filename = '{:05}.t7'.format(engine.state.epoch))


tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters = False)
tb_logger.start_server(tensorboard_port)

trainer.run(train_loader, max_epochs = max_epochs)



