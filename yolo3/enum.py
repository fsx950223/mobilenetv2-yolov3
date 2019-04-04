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