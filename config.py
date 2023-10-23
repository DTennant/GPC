# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/home/zbc/datasets/cifar10'
cifar_100_root = '/home/zbc/datasets/cifar100'
cub_root = '/disk/scratch_fast/bingchen/ssb/cub/'
aircraft_root = '/data/user-data/fgvc/aircraft/fgvc-aircraft-2013b'

inat_mini_root = '/data/zbc-data/inat/'
domainnet_root = '/data/zbc-data/domainnet/'
color_symbol_root = '/home/zbc/universal-category-discovery/dsprites-dataset/color_symbol/'

# not supported yet
herbarium_dataroot = '/work/sagar/datasets/herbarium_19/'
imagenet_root = '/data/user-data/imagenet'

# OSR Split dir
osr_split_dir = '/afs/inf.ed.ac.uk/user/s23/s2329503/GPC-shared/data/ssb_splits/'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/disk/scratch_fast/bingchen/cache_root/torch/hub/checkpoints/dino_vitbase16_pretrain.pth'
mae_inat_pretrain_path = '/home/zbc/mae/output_dir_inat_mini/ep400/checkpoint-399.pth'
feature_extract_dir = '/home/zbc/universal-category-discovery/ext_features'     # Extract features to this directory
exp_root = '/home/zbc/universal-category-discovery/dev_outputs/inat_mini/'          # All logs and checkpoints will be saved here


# -----------------
# YACS Configs
# -----------------
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = 'resnet50'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.LABEL_SMOOTH = False
_C.MODEL.PRETRAIN_PATH = ''
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.0
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.0
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root PATH to the dataset
_C.DATASETS.DATA_PATH = '/home/zbc/data/market1501/'
# PATH to train set
_C.DATASETS.TRAIN_PATH = 'bounding_box_train'
# PATH to query set
_C.DATASETS.QUERY_PATH = 'query'
# PATH to gallery set
_C.DATASETS.GALLERY_PATH = 'bounding_box_test'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.FP16 = False

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# Whether or not use Cython to eval
_C.SOLVER.CYTHON = True

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.WEIGHT = ""
_C.TEST.DEBUG = False
_C.TEST.MULTI_GPU = False
_C.TEST.RERANK = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""

# Alias for easy usage
cfg = _C
