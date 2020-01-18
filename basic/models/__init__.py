from basic.models.scannet import Scannet,scannet_out
from basic.models.deeplab import *
import torchvision.models as models
# from torchvision import models
inception=models.inception_v3(num_classes=2)
#inception.fc = nn.Linear(2048,2)



MODELS = {
    'scannet': Scannet(),
    'inception3':inception,
    "DeepLabResNet":deepresnet
}
OUT_FN={
    'scannet':scannet_out,
    "DeepLabResNet":None
}