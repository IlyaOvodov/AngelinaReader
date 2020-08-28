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
    model_name = 'NN_results/all_data_{model_params.encoder_params.aspect_ratios[0]}_{model_params.loss_params.class_loss_scale}_{augmentation.rotate_limit}',
    data = AttrDict(
        get_points = False,
        class_as_6pt=False,    # классификация присутствия каждой точки в рамке отдельно
        batch_size = 12,
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margions to char width
        max_std = 0.1,
        train_list_file_names = [
            r'DSBI/data/my_train.txt', #0.5),
            r'My/labeled/labeled2/train_books.txt',
            r'My/labeled/labeled2/train_withtext.txt',
            r'My/labeled/labeled2/train_pupils.txt',
            r'My/labeled/labeled2/train_pupils.txt',
            r'My/labeled/not_braille/_not_braille.txt',
            r'My/labeled/ASI/student_book_p1.txt',
            r'My/labeled/ASI/turlom_c2.txt', #3),
            r'My/labeled/ASI/turlom_c2.txt',
            r'My/labeled/ASI/turlom_c2.txt',
            r'web_uploaded/re-processed200823.txt',
            r'ASI_results/braile_photos_and_scans.txt',
        ],
        val_list_file_names = {
            # 'val_dsbi' :  [
            #                r'DSBI/data/my_val1.txt',
            #                r'DSBI/data/my_val1.txt',
            #               ],
            'val_2' : [
                            r'My/labeled/labeled2/val_books.txt',
                            r'My/labeled/labeled2/val_withtext.txt',
                            r'My/labeled/labeled2/val_pupils.txt',
                           ],
            'val_3_asi': [
                r'My/labeled/ASI/turlom_c15_photo_1p.txt',
                r'My/labeled/ASI/turlom_c15_photo_1p.txt',
                r'My/labeled/ASI/turlom_c15_photo_1p.txt',
            ],
            'val_3_asi_scan': [
                r'My/labeled/ASI/turlom_c15_scan_1p.txt',
                r'My/labeled/ASI/turlom_c15_scan_1p.txt',
                r'My/labeled/ASI/turlom_c15_scan_1p.txt',
            ],
        }
    ),
    augmentation = AttrDict(
        img_width_range=(614, 1840),  # 768*0.8, 1536*1.2
        stretch_limit = 0.1,
        rotate_limit = 5,
    ),
    model = 'retina',
    model_params = AttrDict(
        encoder_params = AttrDict(
            #anchor_areas = [5*5., 6*6., 10*10.,],
            anchor_areas=[8*16., 12*24., 16*32.,], # [22*22*0.62, 33*33*0.62, 45*45*0.62,], #[8*16., 12*24., 16*32.,],
            aspect_ratios=[1 / 2.,],  # [0.62,], #[1 / 2.,],
            #aspect_ratios=[1.,],
            #scale_ratios=[1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
            iuo_fit_thr = 0, # if iou > iuo_fit_thr => rect fits anchor
            iuo_nofit_thr = 0,
        ),
        loss_params=AttrDict(
            class_loss_scale = 100,
        ),
    ),
    load_model_from = 'NN_results/retina_chars3_0.5_100_5_090399/models/clr.009.t7',  # retina_chars_d58e5f # retina_chars_7e1d4e
    optim = 'torch.optim.SGD',
    optim_params = AttrDict(
        lr=0.0001,
        momentum=0.9,
        #weight_decay = 0, #0.001,
        #nesterov = False,
    ),
    lr_finder=AttrDict(
        iters_num=200,
        log_lr_start=-4,
        log_lr_end=-0.3,
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
        min_lr=1e-4,
        max_lr=0.03,
        period_epochs=500,
        scale_max_lr=0.95,
        scale_min_lr=0.95,
    ),
)