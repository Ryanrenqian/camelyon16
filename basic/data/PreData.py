import os
import sys
import glob
import numpy as np
import argparse
import logging


def filter(otsu, x, y, gap, threshold):
    '''
    set threshhold to filter useless OTSU region
    '''
    count = np.sum(otsu[int(y-gap*0.5):int(y + gap*0.5), int(x+0.5*gap):int(x + gap+0.5*gap)])
    return (count / (gap * gap)) < threshold



def generate_patch_from_slide(slide_basename, normal_file,tumor_file ,otsu_dict, gt_mask_dict, threshold,image_size=244,down_sample=64):
    """
    边扫描边生成新的dataset
    """

    slide_otsu = otsu_dict[slide_basename]
    x, y = np.where(slide_otsu > 0) #
#     xy = [[x[i], y[i]] for i in range(len(x))]

    tumor_count, normal_count = 0, 0
    try:
        slide_gt_mask = gt_mask_dict[slide_basename]
        slide_is_tumor = True
    except:
        slide_is_tumor = False
    gap= image_size /down_sample
    for _x,_y in zip(x, y):
        if filter(slide_otsu, _x, _y, gap,threshold): # 过滤掉低于阈值的区域
            continue
        level0_x, level0_y = int((_y+0.5) * down_sample), int((_x+0.5) * down_sample) # slide 的坐标
        if slide_is_tumor:
            if slide_gt_mask[_x, _y] > 0:
                tumor_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                tumor_count += 1
            else:
                normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                normal_count += 1
        else:
            normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
            normal_count +=1
    return tumor_count,normal_count

def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False

def preData(**kwargs):
    otsu_folder = kwargs["otsu_folder"]
    gt_mask_folder = kwargs["gt_mask_folder"]
    save_path = kwargs["save_path"]
    downsample = kwargs["downsample"]
    otsu_dict = {}
    otsu_list = glob.glob(os.path.join(otsu_folder, '*.npy'))
    otsu_list.sort()
    total_point = 0
    tumor_count,normal_count=0, 0
    normal_file=os.path.join(save_path,'train_normal.list')
    f_n = open(normal_file,'w')
    tumor_file=os.path.join(save_path,'train_tumor.list')
    f_t = open(tumor_file,'w')
    # reading otsu
    for otsu in otsu_list:
        # 剔除部分数据，tumor_114
        if skip_slide(otsu):
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        otsu_dict[_basename] = np.load(otsu)
        x, y = np.where(otsu_dict[_basename] > 0)
        total_point += len(x)
    logging.info(f'read OSTU: {total_point}')
    # raading gt mask
    gt_mask_list = glob.glob(os.path.join(gt_mask_folder, '*.npy'))
    gt_mask_dict = {}
    for gt_mask in gt_mask_list:
        # 剔除部分数据，tumor114
        if skip_slide(gt_mask):
            continue
        _basename = os.path.basename(gt_mask).split('_resize')[0]
        gt_mask_dict[_basename] = np.load(gt_mask)
    logging.info(f'read GT mask')
    logging.info('Generate train_dataset')
    for i in range(len(otsu_list)):
        otsu = otsu_list[i]
        if skip_slide(otsu):
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        add_t,add_n = generate_patch_from_slide(slide_basename=_basename, normal_file=f_n, tumor_file=f_t,otsu_dict=otsu_dict, gt_mask_dict=gt_mask_dict, threshold=kwargs["threshold"],down_sample=downsample)
        tumor_count += add_t
        normal_count += add_n
    logging.info(f'Tumor:{tumor_count}\tNormal:{normal_count}')
    f_n.close()
    f_t.close()
    return tumor_file,normal_file
