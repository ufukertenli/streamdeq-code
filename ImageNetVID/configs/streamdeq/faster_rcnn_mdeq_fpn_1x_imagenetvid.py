_base_ = [
    '../_base_/models/faster_rcnn_mdeq_fpn.py',
    '../_base_/datasets/imagenet_vid_dataset_mdeq.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
runner = dict(type='EpochBasedRunner', max_epochs=7)
evaluation = dict(interval=7, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=1)
log_config = dict(interval=100)
