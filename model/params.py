import local_config
from ovotools import AttrDict

settings = AttrDict(
    max_epochs=100000,
    tensorboard_port=6006,
    device='cuda:0',
    findLR=False,
    can_overwrite=True,
)

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/retina_DSBI_6pt',
    data = AttrDict(
        get_points = False,
        class_as_6pt=True,    # классификация присутствия каждой точки в рамке отдельно
        batch_size = 12,
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
        img_width_range=( 614, 1840, ),  # 768*0.8, 1536*1.2
        stretch_limit = 0.1,
        rotate_limit = 5,
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
        lr=0.0001,
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
       scale_max_lr=0.98,
       scale_min_lr=0.98,
    ),
)