from basic.models.scannet import Scannet,scannet_out
from basic.models.deeplab import *


MODELS = {
    'scannet': Scannet(),
    "DeepLabMobileNet":DeepLab(backbone='mobilenet', output_stride=2,num_classes=2),
    # "DeepLabResNet":DeepLab(backbone='resnet', output_stride=2,num_classes=2)
}
OUT_FN={
    'scannet':scannet_out
}