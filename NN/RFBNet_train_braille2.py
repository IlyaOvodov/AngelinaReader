import sys
sys.path.append(r'..')
import local_config

from ovotools import AttrDict

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/RFBNet0xxxxxxxxx',
    data = AttrDict(
        batch_size = 100,
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        net_hw = (300,300)
    ),
    prior_box = AttrDict( {
        'feature_maps': [19, 10, 5, 3, 2, 1],
        'min_dim': 300,
        'steps': [16, 32, 64, 100, 150, 300],
        'min_sizes': [45, 90, 135, 180, 225, 270],
        'max_sizes': [90, 135, 180, 225, 270, 315],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
    }),
    load_model_from = 'NN_results/RFBNet0_db0807.model/00500.t7',
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=1e-5,# momentum=0.9,
        #weight_decay=1e-3
    ),
)
max_epochs = 10000
device = 'cuda:0'

params.save(can_overwrite = True)

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools

import DSBI_invest.data
import my_models

train_loader, (val_loader1, val_loader2) = DSBI_invest.data.create_dataloaders(params)

model, loss = my_models.create_model(params, 'train')

optimizer = eval(params.optim)(model.parameters(), **params.optim_params)

metrics = {
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: len(y)),
    'loss.loss_l': ignite.metrics.Loss(loss.get_loss_l, batch_size=lambda y: len(y)),
    'loss.loss_c': ignite.metrics.Loss(loss.get_loss_c, batch_size=lambda y: len(y)),
}
metrics_ev = {
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: len(y)),
    'loss.loss_l': ignite.metrics.Loss(loss.get_loss_l, batch_size=lambda y: len(y)),
    'loss.loss_c': ignite.metrics.Loss(loss.get_loss_c, batch_size=lambda y: len(y)),
}

trainer = ovotools.ignite_tools.create_supervised_trainer(model, optimizer, loss, metrics=metrics, device = device)
evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics_ev, device = device)

best_model_buffer = ovotools.ignite_tools.BestModelBuffer(model, 'train:loss', params, save_to_dir_suffix = '.model', verbose = 0)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = {'val': val_loader1, 'val2': val_loader2},
                                                                best_model_buffer=best_model_buffer,
                                                                params = params)
trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, log_training_results, event = ignite.engine.Events.EPOCH_COMPLETED)

#clr_scheduler = ovotools.ignite_tools.ClrScheduler(train_loader, model, optimizer, 'train:loss', params, engine = trainer)

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})

@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def save_model(engine):
    if engine.state.epoch % 500 == 0:
        best_model_buffer.save_model('{:05}'.format(engine.state.epoch))

tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters = False)

trainer.run(train_loader, max_epochs = max_epochs)



