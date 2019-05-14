from enum import Enum,unique

@unique
class OPT(Enum):
    NORMAL = 0
    XLA = 1
    DEBUG = 2
    MKL = 3

@unique
class BACKBONE(Enum):
    MOBILENETV2=0
    INCEPTION_RESNET2=1
    DENSENET=2
    DARKNET53=3

@unique
class BOX_LOSS(Enum):
    MSE=0
    GIOU=1

@unique
class DATASET_MODE(Enum):
    TRAIN=0
    VALIDATE=1
    TEST=2