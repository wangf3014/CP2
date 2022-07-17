norm_cfg = dict(type='BN', requires_grad=True)
pretrain_path = None    # Please set the path to pretrained weights for Quick Tuning

model = dict(
    type='EncoderDecoder',
    pretrained=pretrain_path,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        contrast=True,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
