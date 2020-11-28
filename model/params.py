import local_config
from ovotools import AttrDict

settings = AttrDict(
    max_epochs=2000,
    tensorboard_port=6006,
    device='cuda:0',
    findLR=False,
    can_overwrite=False,
    regular_save_period = (500, 1),
)

pseudo_step = '1'
pseudo_opt = '1'

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/pseudo'+pseudo_step+'.'+pseudo_opt+'_scores-{data.scores_filter[0][1]}-{data.scores_filter[1][1]}_ignore-{model_params.encoder_params.ignored_scores[0]}-{model_params.encoder_params.ignored_scores[1]}',
    data = AttrDict(
        get_points = False,
        class_as_6pt=False,    # классификация присутствия каждой точки в рамке отдельно
        #load_front_side=True,    # load recto side
        #load_reverse_side=True,  # load revers side
        batch_size = 12,
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margions to char width
        max_std = 0.1,
        train_list_file_names = [
            r'DSBI/data/train_li2.txt',
            r'DSBI/data/val_li2.txt',
            r'AngelinaDataset/pseudo/step_'+pseudo_step+'_opt_'+pseudo_opt+'/books/train.txt',
            r'AngelinaDataset/pseudo/step_'+pseudo_step+'_opt_'+pseudo_opt+'/handwritten/train.txt',
            r'AngelinaDataset/not_braille/train.txt',
        ],
        val_list_file_names = {
            #'test' :  [r'DSBI/data/test_li2.txt',],
            'books': [
                 r'AngelinaDataset/books/val.txt',
                 #r'AngelinaDataset/handwritten/val.txt',
            ],
            'hand': [
                 #r'AngelinaDataset/books/val.txt',
                 r'AngelinaDataset/handwritten/val.txt',
            ],
            # 'two': [
            #      r'AngelinaDataset/books/val.txt',
            #      r'AngelinaDataset/handwritten/val.txt',
            # ],
            'DSBI': [
                r'DSBI/data/test_li2.txt',
            ]
        },
        scores_filter=((5, 0.64), (25, 0.81)),  # quantile % : score_threshold
        target_metric='books:f1',
    ),
    augmentation = AttrDict(
        img_width_range=( 550, 1150, ),  # 768*0.8, 1536*1.2  ,550, 1150,   810, 890
        stretch_limit = 0.1,
        rotate_limit = 5,
        #blur_limit = 0,
        #RandomBrightnessContrast = False,
        #JpegCompression = False,
    ),
    model = 'retina',
    model_params = AttrDict(
        num_fpn_layers=4,
        encoder_params = AttrDict(
            fpn_skip_layers=1,
            anchor_areas=[ 34*55/4, ], # [22*22*0.62, 33*33*0.62, 45*45*0.62,], #[8*16., 12*24., 16*32.,], # 34*55/4
            aspect_ratios=[0.62,],  # [0.62,], #[1 / 2.,],
            scale_ratios=[1.],
            iuo_fit_thr = 0, # if iou > iuo_fit_thr => rect fits anchor
            iuo_nofit_thr = 0,
            ignored_scores = (0.35,0.81),
        ),
        loss_params=AttrDict(
            class_loss_scale = 100,
        ),
    ),
    load_model_from = 'NN_results/pseudo3.3_scores-0.67-0.77_ignore-0.25-0.77_05091c/models/best.t7',  # retina_chars_d58e5f # retina_chars_7e1d4e
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=0.00001,
        #momentum=0.9,
        #weight_decay = 0, #0.001,
        #nesterov = False,
    ),
    lr_finder=AttrDict(
        iters_num=200,
        log_lr_start=-5,
        log_lr_end=-1,
    ),
    lr_scheduler=AttrDict(
        type='MultiStepLR', #'ReduceLROnPlateau', 'MultiStepLR'
        params=AttrDict(
            # MultiStepLR:
            milestones=[550, 1050, 1550],
            gamma=0.1,
        #     # ReduceLROnPlateau:
        #     mode='max',
        #     factor=0.1,
        #     patience=1000,
        ),
    ),
    clr=AttrDict(
        warmup_epochs=10,
        min_lr=1e-5,
        max_lr=0.0001,
        period_epochs=500,
        scale_max_lr=0.90,
        scale_min_lr=0.90,
    ),
)