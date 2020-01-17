from basic.data.datasets import *
from basic.data.sampler import RandomSampler
from basic.data.DataLoader import *
from basic.data.PreData import *
Sampler={
    "RandomSampler":RandomSampler
}
DATALOADER={
    'LoaderOne':LoaderOne,
    'DynamicLoader':DynamicLoader,
}