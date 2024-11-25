import local_config
from ovotools import AttrDict

settings = AttrDict(
    max_epochs=20000,
    tensorboard_port=6006,
    device='cuda:0',
    findLR=False,
    can_overwrite=False,
    eval_period=1,
    regular_save_period = (500, 1),
)

pseudo_step = '1'
pseudo_opt = '1'

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/241009_CenterNet/dla169_S_5cb2fa_2_asi1',
    data = AttrDict(
        get_points = False,
        class_as_6pt=False,    # классификация присутствия каждой точки в рамке отдельно
        #load_front_side=True,    # load recto side
        #load_reverse_side=True,  # load revers side
        batch_size = 28,
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margions to char width
        max_std = 0.1,
        train_list_file_names = [
            r'DSBI/data/train_li2.txt',
            r'DSBI/data/val_li2.txt',
            r'AngelinaDataset/books/train.txt',
            r'AngelinaDataset/handwritten/train.txt',
            r'AngelinaDataset/not_braille/train.txt',
            'ASI/student_book_p1.txt',
            ( 'ASI/turlom_c2.txt', 3, ),
            # ( 'web_uploaded/re-processed200823.txt', 0.125, {'calc_cls': False,}, ),
            # ( 'ASI_results/braile_photos_and_scans.txt', 1, {'calc_cls': False,}, ),
        ],
        val_list_file_names = {
            'ang2' :  [r'AngelinaDataset/uploaded/test.txt',],
            # 'books': [
            #      r'AngelinaDataset/books/val.txt',
            #      #r'AngelinaDataset/handwritten/val.txt',
            # ],
            # 'hand': [
            #      #r'AngelinaDataset/books/val.txt',
            #      r'AngelinaDataset/handwritten/val.txt',
            # ],
            # 'ang': [
            #      r'AngelinaDataset/books/val.txt',
            #      r'AngelinaDataset/handwritten/val.txt',
            # ],
            # 'DSBI': [
            #     r'DSBI/data/test_li2.txt',
            # ]
        },
        #scores_filter=((5, 0.64), (25, 0.81)),  # quantile % : score_threshold
        target_metric='ang2:metrics.f1',
    ),
    augmentation = AttrDict(
        img_width_range=( 614, 1840, ),  # 768*0.8, 1536*1.2  ,550, 1150,   810, 890
        stretch_limit = 0.1,
        rotate_limit = 5,
        #blur_limit = 0,
        #RandomBrightnessContrast = False,
        #JpegCompression = False,
        VerticalFlip=False,
    ),
    model = 'centernet',
    model_params = AttrDict(
        center_net_path = '/home/ovod/file_server/pub_data/Research/3rd_party/CenterNet',
        arch = 'dlav0_169',
        use_hm1 = True,
        # hm_weight = 0,
    ),
    load_model_from = '/home/ovod/file_server/pub_data/BrailleData/NN_results/241009_CenterNet/v1_base_data_hm1_asi1_dla169_S_5cb2fa/models/best.t7',
    optim = 'torch.optim.SGD',
    optim_params = AttrDict(
        lr=0.001,
        momentum=0.9,
        weight_decay = 0.001, # 0, #0.001,
        #nesterov = False,
    ),
    loss_optim_params = AttrDict(
        lr=0.001,
    ),
    lr_finder=AttrDict(
        iters_num=200,
        log_lr_start=-5,
        log_lr_end=-1,
    ),
    lr_scheduler=AttrDict(
        # type='clr', #'ReduceLROnPlateau', 'MultiStepLR'
        type='MultiStepLR',
        params=AttrDict(
            # MultiStepLR:
            milestones=[500, 1000, 1500],
            gamma=0.1,
        #     # ReduceLROnPlateau:
        #     mode='max',
        #     factor=0.1,
        #     patience=1000,
        ),
    ),
    clr=AttrDict(
        warmup_iters=100,
        min_lr=1e-4,
        max_lr=0.01,
        period_iters=10000,
        scale_max_lr=0.2,
        scale_min_lr=0.2,
    ),
)
