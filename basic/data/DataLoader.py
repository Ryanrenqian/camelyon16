import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from basic.data import DATASET,Sampler
import torchvision.transforms as transforms
import torch
import torch.utils.data
import random
class RandomScale(object):
    def __init__(self, shorter_side_range):
        self.shorter_side_range = shorter_side_range

    def __call__(self, img):
        shorter_side_scale = random.uniform(*self.shorter_side_range) / min(img.size)
        new_size = [round(shorter_side_scale * img.width), round(shorter_side_scale * img.height)]
        img = img.resize(new_size)
        return img


class MultiViewTenCrop(object):
    def __init__(self, multi_view, size=224, vertical_flip=False):
        self.multi_view = multi_view
        self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        img_list = []
        for view in self.multi_view:
            img_view = RandomScale((view, view))(img)
            img_ten = transforms.TenCrop(self.size)(img_view)
            img_list = img_list + list(img_ten)
        return transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize((0.837, 0.584, 0.706), (0.141, 0.232, 0.184))(transforms.ToTensor()(crop)) for crop in
             crops]))(img_list)


class DataLoader(object):
    def __init__(self):
        self.dataset=None

    def load_data(self):
        pass
    def prepare_data(self):
        pass

    def get_transforms(self,shorter_side_range=(224, 224), size=(224, 224)):
        return transforms.Compose([RandomScale(shorter_side_range=shorter_side_range),
                                   transforms.RandomCrop(size=size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class DynamicLoader(DataLoader):
    def __init__(self,**kwargs):
        super(DynamicLoader,self).__init__()
        _size=kwargs["crop_size"]
        self.dataset=DATASET["DynamicDataset"](tumor_list=kwargs["tumor_list"],
                                                     normal_list=kwargs["normal_list"],
                                                     transform=self.get_transforms(shorter_side_range = (_size, _size), size = (_size, _size)),
                                                     data_size=kwargs["data_size"],
                                                     tif_folder=kwargs["tif_folder"],
                                                     patch_size=kwargs["patch_size"])
        if kwargs["sampler"] in Sampler.keys():
            self.sampler = Sampler[kwargs["sampler"]](self.dataset, kwargs["data_size"])

    def load_data(self,**kwargs):
        return torch.utils.data.DataLoader(self.dataset, batch_size=kwargs["batch_size"],
                                           sampler=self.sampler, num_workers=kwargs["num_workers"])

class ValidDataLoader(DataLoader):
    def __init__(self,**kwargs):
        super(DataLoader,self).__init__()
        _size = kwargs["crop_size"]
        # transform = self.get_transforms(shorter_side_range=(_size, _size), size=(_size, _size))
        self.dataset = DATASET["ValidDataset"](tumor_list=kwargs["tumor_list"],
                                                     normal_list=kwargs["normal_list"],
                                                     # transform=transform,
                                                     tif_folder=kwargs["tif_folder"],
                                                     patch_size=kwargs["patch_size"])

    def load_data(self,**kwargs):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=kwargs["batch_size"],
                                           num_workers=kwargs["num_workers"])

class HardDataLoader(DataLoader):
    def __init__(self,hardlist,**kwargs):
        super(DataLoader,self).__init__()
        _size = kwargs["crop_size"]
        transform = self.get_transforms(shorter_side_range=(_size, _size), size=(_size, _size))
        self.dataset = DATASET["ListDataset"](hardlist,
                                              transform=transform,
                                              all_class=0,
                                              tif_folder=kwargs["tif_folder"],
                                              patch_size=kwargs["patch_size"])

    def load_data(self,**kwargs):
        return torch.utils.data.DataLoader(self.dataset, batch_size=kwargs["batch_size"],
                                            num_workers=kwargs["num_workers"])


