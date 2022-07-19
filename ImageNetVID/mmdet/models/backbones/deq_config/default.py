# Modified based on the HRNet repo.

from yacs.config import CfgNode as CN

_C = CN()

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'mdeq'  # Default for classification
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.WNORM = False
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_LAYERS = 5
_C.MODEL.NUM_GROUPS = 4
_C.MODEL.DROPOUT = 0.0
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DOWNSAMPLE_TIMES = 2
_C.MODEL.EXPANSION_FACTOR = 5
_C.MODEL.BLOCK_GN_AFFINE = True
_C.MODEL.FUSE_GN_AFFINE = True
_C.MODEL.POST_GN_AFFINE = True
_C.MODEL.CONV_INJECTION = False
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

# DEQ related
_C.DEQ = CN()
_C.DEQ.F_SOLVER = 'broyden'
_C.DEQ.B_SOLVER = 'broyden'
_C.DEQ.STOP_MODE = 'abs'
_C.DEQ.RAND_F_THRES_DELTA = 2
_C.DEQ.F_THRES = 30
_C.DEQ.B_THRES = 40
_C.DEQ.SPECTRAL_RADIUS_MODE = False
_C.DEQ.UNROLL = False
_C.DEQ.STOCH = False

_C.LOSS = CN()
_C.LOSS.JAC_LOSS_FREQ = 0.0
_C.LOSS.JAC_LOSS_WEIGHT = 0.0
_C.LOSS.JAC_INCREMENTAL = int(1e8)
_C.LOSS.JAC_STOP_ITER = int(1e8)  # The epoch at which we stop applying Jacobian regularization loss
_C.LOSS.PRETRAIN_JAC_LOSS_WEIGHT = 0.0
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = True
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# train
_C.TRAIN = CN()
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.CLIP = -1.0
_C.TRAIN.PRETRAIN_STEPS = 0


def update_config(cfg, exp_cfg):
    cfg.defrost()
    cfg.merge_from_file(exp_cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
