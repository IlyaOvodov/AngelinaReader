import local_config
from ovotools import AttrDict

settings = AttrDict(
    max_epochs=100000,
    tensorboard_port=6006,
    device='cuda:0',
    findLR=False,
    can_overwrite=False,
)

params = AttrDict(
    data_root = local_config.data_path,
    model_name = 'NN_results/angelina_lay{model_params.num_fpn_layers}_{model_params.loss_params.class_loss_scale}',
    data = AttrDict(
        get_points = False,
        class_as_6pt=False,    # классификация присутствия каждой точки в рамке отдельно
        #load_front_side=True,    # load recto side
        #load_reverse_side=True,  # load revers side
        batch_size = 24,
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margions to char width
        max_std = 0.1,
        train_list_file_names = [
            r'DSBI/data/train_li2.txt',
            r'DSBI/data/val_li2.txt',
            r'AngelinaDataset/books/train_books.txt',
            r'AngelinaDataset/handwritten/train_handwritten.txt',
            r'AngelinaDataset/not_braille/train_not_braille.txt',
        ],
        val_list_file_names = {
            'dsbi' :  [r'DSBI/data/test_li2.txt',],
            'Angelina': [
                r'AngelinaDataset/books/val_books.txt',
                r'AngelinaDataset/handwritten/val_handwritten.txt',
            ]
        }
    ),
    augmentation = AttrDict(
        img_width_range=( 550, 1150, ),  # 768*0.8, 1536*1.2
        stretch_limit = 0.1,
        rotate_limit = 5,
        #blur_limit = 0,
        #RandomBrightnessContrast = False,
        #JpegCompression = False,
    ),
    model = 'retina',
    model_params = AttrDict(
        num_fpn_layers=3,
        encoder_params = AttrDict(
            anchor_areas=[ 18*28, ], # [22*22*0.62, 33*33*0.62, 45*45*0.62,], #[8*16., 12*24., 16*32.,], # 34*55/4
            aspect_ratios=[0.62,],  # [0.62,], #[1 / 2.,],
            scale_ratios=[1.],
            iuo_fit_thr = 0, # if iou > iuo_fit_thr => rect fits anchor
            iuo_nofit_thr = 0,
        ),
        loss_params=AttrDict(
            class_loss_scale = 1,
        ),
    ),
    #load_model_from = 'NN_results/angelina_lay3_1_pt_c4846a/models/clr.002.t7',  # retina_chars_d58e5f # retina_chars_7e1d4e
    optim = 'torch.optim.Adam',
    optim_params = AttrDict(
        lr=0.0001,
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
        type='clr',
        #params=AttrDict(
        #    milestones=[5000, 10000,],
        #    gamma=0.1,
        #),
    ),
    clr=AttrDict(
        warmup_epochs=10,
        min_lr=1e-5,
        max_lr=0.0001,
        period_epochs=500,
        scale_max_lr=0.95,
        scale_min_lr=0.90,
    ),
)