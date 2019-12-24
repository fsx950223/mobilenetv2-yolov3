from enum import Enum, unique


@unique
class MODE(Enum):
    TRAIN = 0
    IMAGE = 1
    VIDEO = 2
    TFLITE = 3
    SERVING = 4
    MAP = 5
    PRUNE = 6
    TFJS = 7
    TRAIN_BACKBONE = 8


@unique
class OPT(Enum):
    XLA = 0
    DEBUG = 1
    MKL = 2


@unique
class BACKBONE(Enum):
    MOBILENETV2 = 0
    EFFICIENTNET = 1
    DARKNET53 = 2


@unique
class BOX_LOSS(Enum):
    MSE = 0
    GIOU = 1


@unique
class DATASET_MODE(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2
