from basic.models.scannet import Scannet,scannet_out
from basic.models.deeplab import *

MODELS = {
    'scannet': Scannet(),
    "DeepLabResNet":deepresnet
}
OUT_FN={
    'scannet':scannet_out,
    "DeepLabResNet":None
}