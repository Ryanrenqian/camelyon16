import os
import sys
import glob
import numpy as np
import argparse
import logging
import random
import tqdm
def filter(otsu, x, y, gap, threshold):
    '''
    set threshhold to filter useless OTSU region
    '''
    count = np.sum(otsu[int(y-gap*0.5):int(y + gap*0.5), int(x+0.5*gap):int(x + gap+0.5*gap)])
    return (count / (gap * gap)) < threshold



def generate_all_patch_from_slide(slide_basename, normal_file,tumor_file ,otsu_dict, gt_mask_dict, image_size=244,down_sample=64):
    """
    抽取所有有用的点用于训练
    """

    slide_otsu = otsu_dict[slide_basename]
    x, y = np.where(slide_otsu > 0)
    tumor_count, normal_count = 0, 0
    try:
        slide_gt_mask = gt_mask_dict[slide_basename]
        slide_is_tumor = True
    except:
        slide_is_tumor = False
    data = random.sample(zip(x, y), len(x) // 10)  # 随机抽取百分10%
    if slide_is_tumor:
        for _x, _y in data:
            level0_x, level0_y = int((_y + 0.5) * down_sample), int((_x + 0.5) * down_sample)  # slide 的坐标
            if slide_gt_mask[_x, _y] > 0:
                tumor_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                tumor_count += 1
            elif normal_count <= 5 * tumor_count:
                normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                normal_count += 1
    else:
        for _x,_y in data:

            level0_x, level0_y = int((_y + 0.5) * down_sample), int((_x + 0.5) * down_sample)  # slide 的坐标
            normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
            normal_count += 1
    return tumor_count,normal_count

def generate_limited_patch_from_slide(slide_basename, normal_file,tumor_file,tumor_size,nomral_size, otsu_dict, gt_mask_dict, image_size=244,down_sample=64):
    """
    边扫描边生成新的dataset
    """

    slide_otsu = otsu_dict[slide_basename]
    x, y = np.where(slide_otsu > 0)
    tumor_count, normal_count = 0, 0
    try:
        tumor_array = gt_mask_dict[slide_basename]
    except:
        tumor_array=np.zeros_like(slide_otsu)
    # 随机抽取肿瘤样本
    x, y = np.where(tumor_array > 0)
    data = [(i, j) for i, j in zip(x, y)]
    random.shuffle(data)  # 随机抽样
    tumor_count,normal_count=0,0
    for x,y in data:
        try:
            tumor_array[x+image_size//down_sample,y+image_size//down_sample]
        except:
            continue
        if tumor_count==tumor_size:
            break
        level0_x, level0_y = int((y + 0.5) * down_sample), int((x + 0.5) * down_sample)
        tumor_file.write(f'{slide_basename}.tif_{level0_x}_{level0_y}\n')
        tumor_count+=1
    normal_array=slide_otsu + (tumor_array*-1)
    x,y = np.where(normal_array==1)
    data = [(i, j) for i, j in zip(x, y)]
    random.shuffle(data)  # 随机抽样
    for x,y in data:
        try:
            tumor_array[x+image_size//down_sample,y+image_size//down_sample]
        except:
            continue
        if normal_count==nomral_size:
            break
        level0_x, level0_y = int((y + 0.5) * down_sample), int((x + 0.5) * down_sample)
        normal_file.write(f'{slide_basename}.tif_{level0_x}_{level0_y}\n')
        normal_count+=1
    return tumor_count,normal_count





def generate_mask_patch_from_slide(slide_basename, patch_list,otsu_dict, gt_mask_dict,num_data=2000,image_size=244,down_sample=64):
    """
    otsu is used for normal patch and mask is made to extract tumor patch.
    :param slide_basename:
    :param normal_file:
    :param tumor_file:
    :param otsu_dict:
    :param gt_mask_dict:
    :param threshold:
    :param image_size:
    :param down_sample:
    :return:
    """

    try:
        nparray = gt_mask_dict[slide_basename] # mask array
    except:
        nparray = otsu_dict[slide_basename] # otsu array
    x_max,y_max = nparray.shape
    x, y = np.where(nparray > 0)
    data=[(i,j) for i,j in zip(x,y)]
    random.shuffle(data) # 随机抽样
    num=0
    for x,y in data:
        try:
            nparray[x+image_size//down_sample,y+image_size//down_sample] # 判断是否对应完整的mask
        except:
            continue
        if num == num_data:
            break
        level0_x,level0_y = int((y + 0.5) * down_sample), int((x + 0.5) * down_sample)
        patch_list.write(f'{slide_basename}.tif_{level0_x}_{level0_y}\n')
        num+=1
    return num

def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False



class GenerateData:
    def __init__(self,**kwargs):
        otsu_folder = kwargs['rawdata']["otsu_folder"]
        gt_mask_folder = kwargs['rawdata']["gt_mask_folder"]
        self.treshold=kwargs.get("threshold",None)
        self.save_path = kwargs['dataset']['train']["save_path"]
        self.image_size=kwargs['predata']["image_size"]
        self.down_sample=kwargs['predata']["down_sample"]
        self.num_data=kwargs['predata']["num_data"]
        self.otsu_dict = {}
        old_otsu_list = glob.glob(os.path.join(otsu_folder, '*.npy'))
        old_otsu_list.sort()
        slide_list=[]
        for otsu in old_otsu_list:
            # 剔除部分数据，tumor_114
            if skip_slide(otsu):
                continue
            _basename = os.path.basename(otsu).split('_resize')[0]
            slide_list.append(_basename)
            self.otsu_dict[_basename] = np.load(otsu)
        self.slide_list=set(slide_list)
        tumor_slides=[]
        logging.info('read OSTU:')
        gt_mask_list = glob.glob(os.path.join(gt_mask_folder, '*.npy'))
        self.gt_mask_dict = {}
        for gt_mask in gt_mask_list:
            # 剔除部分数据，tumor114
            if skip_slide(gt_mask):
                continue
            _basename = os.path.basename(gt_mask).split('_resize')[0]
            tumor_slides.append(_basename)
            self.gt_mask_dict[_basename] = np.load(gt_mask)
        self.tumor_slides=set(tumor_slides)
        logging.info('read GT mask')


    def preAll(self):
        logging.info('Generate train_dataset')
        tumor_count, normal_count = 0, 0
        normal_file = os.path.join(self.save_path, 'train_normal.list')
        f_n = open(normal_file, 'w')
        tumor_file = os.path.join(self.save_path, 'train_tumor.list')
        f_t = open(tumor_file, 'w')
        for otsu in tqdm.tqdm(self.slide_list):
            _basename = os.path.basename(otsu).split('_resize')[0]
            add_t, add_n = generate_all_patch_from_slide(slide_basename=_basename, normal_file=f_n, tumor_file=f_t,
                                                     otsu_dict=self.otsu_dict, gt_mask_dict=self.gt_mask_dict,
                                                    down_sample=self.down_sample)
            tumor_count += add_t
            normal_count += add_n
        logging.info(f'Tumor:{tumor_count}\tNormal:{normal_count}')
        f_n.close()
        f_t.close()
        return tumor_file, normal_file

    def preMask(self):
        logging.info('Generate train_dataset')
        mask_file = os.path.join(self.save_path,"mask.list")
        f=open(mask_file,'w')
        num=0
        for otsu in tqdm.tqdm(self.slide_list):
            slide_basename = os.path.basename(otsu).split('_resize')[0]
            num+=generate_mask_patch_from_slide(slide_basename=slide_basename,
                                                patch_list=f,
                                                otsu_dict=self.otsu_dict,
                                                gt_mask_dict=self.gt_mask_dict,
                                                num_data=self.num_data,
                                                image_size=self.image_size,
                                                down_sample=self.down_sample)
        f.close()
        logging.info(f"extract {num} patches for training")
        return num,mask_file

    def preLimited(self,datasize):
        logging.info('Generate train_dataset')
        tumor_count, normal_count = 0, 0
        normal_file = os.path.join(self.save_path, 'train_normal.list')
        f_n = open(normal_file, 'w')
        tumor_file = os.path.join(self.save_path, 'train_tumor.list')
        f_t = open(tumor_file, 'w')
        normal_remain=datasize/2
        tumor_remain=datasize/2
        while(tumor_remain!=0 && normal_remain !=0):
            tumor_slide_len = len(self.tumor_slides)
            slide_len = len(self.slide_list)
            for basename in tqdm.tqdm(self.slide_list):
                normal_size=normal_remain//slide_len
                tumor_size = 0
                if basename in self.tumor_slides:
                    tumor_size=tumor_remain//tumor_slide_len
                    tumor_slide_len-=1
                add_t, add_n = generate_limited_patch_from_slide(slide_basename=basename, normal_file=f_n, tumor_file=f_t,
                                                         otsu_dict=self.otsu_dict, gt_mask_dict=self.gt_mask_dict,
                                                         down_sample=self.down_sample,nomral_size=normal_size,tumor_size=tumor_size)
                tumor_remain -= add_t
                normal_remain -= add_n
                tumor_count += add_t
                normal_count += add_n
                slide_len-=1
        logging.info(f'Tumor:{tumor_count}\tNormal:{normal_count}')
        f_n.close()
        f_t.close()
        return tumor_file, normal_file
