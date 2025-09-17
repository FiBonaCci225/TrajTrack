
_base_ = '../../default_runtime.py'
data_dir = '/home/ubuntu/os/kitti_tracking/training'
category_name = 'Car'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True
use_rot = False

model = dict(
    type='M2TRACK',
    config=dict(
        bb_scale=1.25,
        bb_offset=2,
        point_sample_size=1024,
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=True,
        use_rot=use_rot,
        motion_cls_seg_weight=0.1,
        use_z=True,
        limit_box=False,
        IoU_space=3,
        center_weight=2,
        angle_weight=10.0,
        seg_weight=0.1,
        bc_weight=1,
    )
)

train_dataset = dict(
    type='MotionTrackingSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Train',
        category_name=category_name,
        preloading=False,
        preload_offset=10
    ),
    config=dict(
        bb_scale=1.25,
        bb_offset=2,
        point_sample_size=1024,
        degrees=False,
        data_limit_box=True,
        motion_threshold=0.15,
        use_augmentation=True,
        num_candidates=4,
        target_thr=None,
        search_thr=None,
        point_cloud_range=point_cloud_range,
        regular_pc=False,
        flip=False,
        box_aware=box_aware
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Test',
        category_name=category_name,
        preloading=False,
        preload_offset=-1
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)