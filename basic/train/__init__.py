from basic.train.train import train_epoch,valid_epoch,hard_epoch
from basic.train.loss import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
Procedure={
    "train":train_epoch,
    "valid":valid_epoch
}
optims={
    "SGD":optim.SGD,
    "Adam":optim.Adam
}
Loss={
    "CrossEntropyLoss":nn.CrossEntropyLoss(),
    "FocalLoss":FocalLoss()
}