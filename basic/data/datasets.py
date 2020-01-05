from  PIL import Image
import torchvision.transforms as transforms
import random,torch
import torch.utils.data as data
import json
import os
import glob
import openslide
import os
import pdb
import time
import numpy as np

class MaskDataset():
    def __init__(self, list_file,mask_folder,transform=None,level=0, tif_folder='/root/workspace/dataset/CAMELYON16/training/*',
                 patch_size=256):
        """
        _patch_list_txt：
        {'xxx.tif_x_y': 0,
         'xxx.tif_x_y': 1,
        } 其中x_y指的是level0上的值
        :param _patch_list_txt:
        :param transform:
        """
        tif_list = glob.glob(os.path.join(tif_folder, '*.tif'))
        tif_list.sort()
        mask_list=glob.glob(os.path.join(mask_folder,'*.npy'))
        with open(list_file,'r')as f:
            self.patch_name_list=f.readlines()
        self.patch_size = patch_size
        self.transform = transform
        self.level=level
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif)
            self.slide_dict[basename] = tif
        self.mask_dict={}
        for mask in mask_list:
            basename = os.path.basename(mask).split('_resize')[0]
            self.mask_dict[basename]=mask

    def __getitem__(self, index):
        patch_name = self.patch_name_list[index].rstrip()
        slide_name = patch_name.split('.tif_')[0] + '.tif'
        slide = openslide.OpenSlide(self.slide_dict[slide_name])  # 直接在这里使用对速度没有明显影响，但slide的缓存会较少很多
        _x, _y = patch_name.split('.tif_')[1].split('_') # 中心点坐标
        _x, _y = int(_x - self.patch_size / 2), int(_y-self.patch_size/2)
        try:
            img = slide.read_region((_x, _y), self.level, [self.patch_size, self.patch_size]).convert('RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        mask=torch.from_numpy(np.load(self.mask_dict[slide_name]))
        return input_img, mask, patch_name

    def __len__(self):
        return len(self.patch_name_list)



class ListDataset():
    def __init__(self, list_file,transform=None,all_class=None,level=0, tif_folder='/root/workspace/dataset/CAMELYON16/training/*',
                 patch_size=256):
        """
        _patch_list_txt：
        {'xxx.tif_x_y': 0,
         'xxx.tif_x_y': 1,
        } 其中x_y指的是level0上的值
        :param _patch_list_txt:
        :param transform:
        """
        tif_list = glob.glob(os.path.join(tif_folder, '*.tif'))
        tif_list.sort()
        with open(list_file,'r')as f:
            self.patch_name_list=f.readlines()
        self.all_class=all_class
        self.patch_size = patch_size
        self.transform = transform
        self.level=level
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif)
            self.slide_dict[basename] = tif

    def __getitem__(self, index):
        if self.all_class==None:
            patch_name,class_id = self.patch_name_list[index].rstrip().split()
        else:
            patch_name = self.patch_name_list[index].rstrip()
            class_id = self.all_class
        slide_name = patch_name.split('.tif_')[0] + '.tif'
        slide = openslide.OpenSlide(self.slide_dict[slide_name])  # 直接在这里使用对速度没有明显影响，但slide的缓存会较少很多
        _x, _y = patch_name.split('.tif_')[1].split('_')
        _x, _y = int(_x), int(_y)
        _x, _y = int(_x - self.patch_size / 2), int(_y - self.patch_size / 2)
        input_img = None
        # print(_x,_y,slide_name)
        try:
            img = slide.read_region((_x, _y), self.level, [self.patch_size, self.patch_size]).convert('RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        # input_img = input_img.cuda()
        return input_img, class_id, patch_name

    def __len__(self):
        return len(self.patch_name_list)

class ValidDataset():
    def __init__(self,normal_list,tumor_list,patch_size,tif_folder='/root/workspace/dataset/CAMELYON16/training/*',transform=None):
        self.tumor = ListDataset(list_file=tumor_list,
                                 tif_folder=tif_folder,
                                 transform=transform,
                                 all_class=1,
                                 patch_size=patch_size)
        self.normal = ListDataset(list_file=normal_list,
                                  tif_folder=tif_folder,
                                  transform=transform,
                                  all_class=0,
                                  patch_size=patch_size)
    @property
    def data(self):
        '''

        :return: Tumor + Normal dataset
        '''
        return data.ConcatDataset([self.tumor,self.normal])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.tumor) + len(self.normal)

    @property
    def shape(self):
        '''

        :return: tumor size, normal size
        '''
        return (len(self.tumor), len(self.normal))

class DynamicDataset():
    '''
    provided Tumor and Normal dataset, Then using RandomSampler for Dynamic Dataset to train model on the fly.
    '''
    def __init__(self,normal_list,tumor_list,transform,patch_size,tif_folder='/root/workspace/dataset/CAMELYON16/training/*'):
        self.tumor = ListDataset(list_file=tumor_list,
                                 tif_folder=tif_folder,
                                 transform=transform,
                                 all_class=1,
                                 patch_size=patch_size)
        self.normal = ListDataset(list_file=normal_list,
                                  tif_folder=tif_folder,
                                  transform=transform,
                                  all_class=0,
                                  patch_size=patch_size)

    @property
    def data(self):
        '''
        :return: Tumor + Normal dataset
        '''
        return data.ConcatDataset([self.tumor,self.normal])

    def  __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.tumor)+len(self.normal)

    @property
    def shape(self):
        '''
        :return: tumor size, normal size
        '''
        return (len(self.tumor),len(self.normal))


