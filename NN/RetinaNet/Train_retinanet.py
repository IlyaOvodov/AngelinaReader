from collections import OrderedDict
import sys
sys.path.append(r'../..')
import local_config

from ovotools import AttrDict

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/retina_DSBI_TEST_noaugm',
    data = AttrDict(
        get_points = False,
        batch_size = 12,
        #mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        #std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margions to char width
        train_list_file_names = [
            r'DSBI\data\train.txt',
            # r'My\labeled\labeled2\train_books.txt',
            # r'My\labeled\labeled2\train_withtext.txt',
            # r'My\labeled\labeled2\train_pupils.txt',
            # r'My\labeled\labeled2\train_pupils.txt',
            # r'My\labeled\not_braille\_not_braille.txt',
        ],
        val_list_file_names = {
            'val_dsbi' :  [
                           r'DSBI\data\test.txt',
                           #r'DSBI\data\my_val2.txt',
                          ],
            # 'val_books' : [
            #                r'My\labeled\labeled2\val_books.txt',
            #                r'My\labeled\labeled2\val_withtext.txt',
            #               ],
            # 'val_pupils' : [
            #                 r'My\labeled\labeled2\val_pupils.txt',
            #                ],
        }
    ),
    augmentation = AttrDict(
        img_width_range=(1024, 1024),  # 768*0.8, 1536*1.2
        stretch_limit = 0,
        rotate_limit = 0,
        blur_limit = 0,
        RandomBrightnessContrast = False,
        JpegCompression = False,
        HorizontalFlip = False,
    ),
    model = 'retina',
    model_params = AttrDict(
        encoder_params = AttrDict(
            #anchor_areas = [5*5., 6*6., 10*10.,],
            anchor_areas = [8*16., 12*24., 16*32.,],
            aspect_ratios=[1 / 2.,],
            #aspect_ratios=[1.,],
            #scale_ratios=[1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
            iuo_fit_thr = 0, # if iou > iuo_fit_thr => rect fits anchor
            iuo_nofit_thr = 0,
        ),
        loss_params=AttrDict(
            # class_loss_scale = 100,
        ),
    ),
    #load_model_from = 'NN_results/retina_chars_7ec096/models/clr.012',
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=0.0002,
        #momentum=0.9,
        #weight_decay = 0, #0.001,
        #nesterov = False,
    ),
    lr_finder=AttrDict(
        iters_num=400,
        log_lr_start=-4,
        log_lr_end=-0.3,
    ),
    lr_scheduler=AttrDict(
        type='clr',
        params=AttrDict(
            milestones=[5000, 10000,],
            gamma=0.1,
        ),
    ),
    clr=AttrDict(
       warmup_epochs=10,
       min_lr=1e-05,
       max_lr=0.0002,
       period_epochs=500,
       scale_max_lr=0.9,
       scale_min_lr=0.9,
    ),
)
max_epochs = 100000
tensorboard_port = 6006
device = 'cuda:0'
findLR = False
can_overwrite = False

if findLR:
    params.model_name += '_findLR'

params.save(can_overwrite = can_overwrite)

import torch
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools
import ovotools.pytorch

import train.data
import create_model_retinanet

ctx = ovotools.pytorch.Context(settings=None, params=params)

model, collate_fn, loss = create_model_retinanet.create_model_retinanet(params, phase='train', device=device)
ctx.net  = model
ctx.loss = loss

train_loader = train.data.create_dataloader(params, collate_fn,
                                            list_file_names=params.data.train_list_file_names, shuffle=True)
val_loaders = { k: train.data.create_dataloader(params, collate_fn, list_file_names=v, shuffle=False)
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

trainer_metrics = {} if findLR else metrics
eval_loaders = {}
if findLR:
    eval_loaders['train'] = train_loader
eval_loaders.update(val_loaders)
eval_event = ignite.engine.Events.ITERATION_COMPLETED if findLR else ignite.engine.Events.EPOCH_COMPLETED
eval_duty_cycle = 2 if findLR else 5
train_epochs = params.lr_finder.iters_num*len(train_loader) if findLR else max_epochs

trainer = ovotools.ignite_tools.create_supervised_trainer(model, ctx.optimizer, loss, metrics=trainer_metrics, device = device)
evaluator = ignite.engine.create_supervised_evaluator(model, metrics=eval_metrics, device = device)

if findLR:
    best_model_buffer = None
else:
    best_model_buffer = ovotools.ignite_tools.BestModelBuffer(ctx.net, 'val_dsbi:loss', minimize=True, params=ctx.params)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = eval_loaders,
                                                                best_model_buffer=best_model_buffer,
                                                                params = params,
                                                                duty_cycles = eval_duty_cycle)
trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

if findLR:
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

@trainer.on(Events.ITERATION_COMPLETED)
def reset_resources(engine):
    engine.state.batch = None
    engine.state.output = None
    #torch.cuda.empty_cache()

trainer.run(train_loader, max_epochs = train_epochs)

