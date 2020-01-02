import os
import sys
import glob
import numpy as np
import json
from skimage import measure
from scipy import ndimage as nd
from random import shuffle
import openslide
import random
import time

import multiprocessing

from random import shuffle
from skimage.measure import points_in_poly
import argparse

import util

'''
    根据otsu的前景和ground truth mask列表，生成patch的坐标list
    生成 patch_list.txt 
        - { 'xxx.tif_x_y': 0,
            'xxx.tif_x_y': 1, }
'''

# 日志
basename = os.path.basename(sys._getframe().f_code.co_filename)
log = util.Log('./log_%s.txt' % basename[:basename.rfind('.')])


def patch_type_in_tumor(slide_gt_mask, x, y, patch_expand):
    """
    a - 前景点为normal
    b - 前景点为normal，但patch size中，部分在tumor区域内
    c - 前景点为tumor
    d - 前景点为tumor，但patch size中，部份在tumor区域外
    """
    x0 = (x - patch_expand) if (x - patch_expand) > 0 else 0
    x1 = (x + patch_expand) if (x + patch_expand) < (slide_gt_mask.shape[0] - 1) else (slide_gt_mask.shape[0] - 1)
    y0 = (y - patch_expand) if (y - patch_expand) > 0 else 0
    y1 = (y + patch_expand) if (y + patch_expand) < (slide_gt_mask.shape[1] - 1) else (slide_gt_mask.shape[1] - 1)
    patch = slide_gt_mask[x0:x1, y0:y1]
    if slide_gt_mask[x, y] == True:
        if patch.sum() == patch.shape[0] * patch.shape[1]:
            return 'c'  # 所有patch点都在tumor
        else:
            return 'd'  # 存在tumor外的点
    else:
        if patch.sum() == 0:
            return 'a'  # 所有patch点都在normal
        else:
            return 'b'  # 存在tumor的点


def generate_patch_from_slide(slide_basename, patch_dict, otsu_dict, gt_mask_dict, total_num=2000, down_sample=64,
                              patch_expand=2):
    """
    在产生点的时候，将abcd的分布也返回
    a - 中心点为normal，且patch包含的点均为normal
    b - 中心点为normal，但patch size中，部分在tumor区域内
    c - 中心点为tumor，且patch包含的点均为tumor
    d - 中心点为tumor，但patch size中，部份在tumor区域外
    """
    patch_type_count = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    slide_otsu = otsu_dict[slide_basename]
    x, y = np.where(slide_otsu > 0)
    xy = [[x[i], y[i]] for i in range(len(x))]
    random.seed(time.time())
    shuffle(xy)
    tumor_count, normal_count = 0, 0
    slide_is_tumor = False
    if slide_basename in gt_mask_dict.keys():
        slide_gt_mask = gt_mask_dict[slide_basename]
        slide_is_tumor = True
    for i in range(len(xy)):
        _x, _y = xy[i]
        # 取patch的时候，以_x,_y像素块的中心为中心。偏差大约32像素点
        level0_x, level0_y = int((_x+0.5) * down_sample), int((_y+0.5) * down_sample)
        if slide_is_tumor:
            if slide_gt_mask[_x, _y] > 0:
                patch_type = patch_type_in_tumor(slide_gt_mask, _x, _y, patch_expand)
                patch_type_count[patch_type] += 1
                if tumor_count < total_num:
                    patch_dict['%s.tif_%04d_%04d' % (slide_basename, level0_x, level0_y)] = 1
                    tumor_count += 1
            else:
                patch_type = patch_type_in_tumor(slide_gt_mask, _x, _y, patch_expand)
                patch_type_count[patch_type] += 1
                if normal_count < total_num:
                    patch_dict['%s.tif_%04d_%04d' % (slide_basename, level0_x, level0_y)] = 0
                    normal_count += 1
        else:
            patch_type_count['a'] += 1
            if normal_count < total_num:
                patch_dict['%s.tif_%04d_%04d' % (slide_basename, level0_x, level0_y)] = 0
                normal_count += 1

        # 抽取数目满足后，停止遍历前景点
        if (not slide_is_tumor and normal_count >= total_num) or (slide_is_tumor and tumor_count >= total_num):
            log(slide_basename, 'tumor and normal patch > %d' % total_num)
            break
        if i == len(xy) - 1:
            log('%s, tumor and normal patch : %04d, %04d' % (slide_basename, tumor_count, normal_count))
    return patch_type_count


def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False


# main parameter
parser = argparse.ArgumentParser(description='generate patch points of WSI')
parser.add_argument('-si', '--start_index', default=0, type=int, help='start index from glob.glob list')
parser.add_argument('-ei', '--end_index', default=0, type=int, help='end index from glob.glob list')
parser.add_argument('-pn', '--patch_number', default=1000, type=int, help='每个slide中提取的最大patch数目')
parser.add_argument('-of', '--otsu_folder',
                    default='/root/workspace/dataset/CAMELYON16/testing',
                    type=str, help='otsu图片列表')
parser.add_argument('-gf', '--gt_mask_folder',
                    default='/root/workspace/dataset/CAMELYON16/training/lesion_annotations',
                    type=str, help='由标注生成的mask')
parser.add_argument('-sp', '--save_path',
                    default='./wsi_coord_generate_save/testing',
                    type=str, help='生成的patch list路径')

if __name__ == '__main__':
    args = parser.parse_args()

    range_start = args.start_index
    otsu_folder = args.otsu_folder
    gt_mask_folder = args.gt_mask_folder
    patch_number = args.patch_number
    save_path = args.save_path  # train.txt  train_hard.txt

    # 数据文件夹准备
    otsu_dict = {}
    otsu_list = glob.glob(os.path.join(otsu_folder, '*.npy'))
    otsu_list.sort()
    total_point = 0
    for otsu in otsu_list:
        # 剔除部分数据，tumor_114
        if skip_slide(otsu):
            log('skip otsu slide: %s' % otsu)
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        otsu_dict[_basename] = np.load(otsu)
        x, y = np.where(otsu_dict[_basename] > 0)
        total_point += len(x)

    gt_mask_list = glob.glob(os.path.join(gt_mask_folder, '*.npy'))
    gt_mask_dict = {}

    for gt_mask in gt_mask_list:
        # 剔除部分数据，tumor114
        if skip_slide(gt_mask):
            log('skip gt_mask slide: %s' % gt_mask)
            continue
        _basename = os.path.basename(gt_mask).split('_resize')[0]
        gt_mask_dict[_basename] = np.load(gt_mask)

    range_end = args.end_index if args.end_index != 0 else len(otsu_list)
    patch_dict = {}
    total_patch_type_count = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    """
        在产生点的时候，将abcd的分布也返回
        a - 中心点为normal，且patch包含的点均为normal
        b - 中心点为normal，但patch size中，部分在tumor区域内
        c - 中心点为tumor，且patch包含的点均为tumor
        d - 中心点为tumor，但patch size中，部份在tumor区域外
    """
    for i in range(range_start, range_end):
        otsu = otsu_list[i % len(otsu_list)]
        if skip_slide(otsu):
            log('skip otsu slide: %s' % otsu)
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        patch_type_count = generate_patch_from_slide(_basename, patch_dict, otsu_dict, gt_mask_dict, patch_number,
                                                     patch_expand=4)
        for k in patch_type_count.keys():
            total_patch_type_count[k] += patch_type_count[k]
        log(patch_type_count, total_patch_type_count)

    log(total_patch_type_count)

    f = open(save_path, 'w')
    f.writelines(json.dumps(patch_dict, indent=4))
    f.close()
