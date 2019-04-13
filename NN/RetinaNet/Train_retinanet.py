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
        lr=1e-4,# momentum=0.9,
        #weight_decay=1e-3
    ),
    decimate_lr_every = 1200,
)
max_epochs = 10000
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
#tb_logger.start_server(tensorboard_port)

trainer.run(train_loader, max_epochs = max_epochs)

