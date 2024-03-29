dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
classes = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
           'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
           'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
           'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
           'watercraft', 'whale', 'zebra')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=False),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'Data/VID',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=[-1,0],
                filter_key_img=True,
                method='uniform',
                return_key_img=True),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            classes=classes,
            load_as_video=False,
            ann_file=data_root + 'annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'Data/DET',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=0,
                filter_key_img=True,
                method='uniform',
                return_key_img=True),
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=[-1, 0],
            filter_key_img=True,
            method='uniform',
            return_key_img=True),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=[-1, 0],
            filter_key_img=True,
            method='uniform',
            return_key_img=True),
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=1, metric='bbox')
