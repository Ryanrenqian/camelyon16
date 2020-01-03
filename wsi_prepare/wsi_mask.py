import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import xml.etree.ElementTree as ET

from scipy import ndimage as nd

import os
import multiprocessing
import threading
import glob
import argparse
import pdb

import util

'''
    issue：提取wsi在32/64(level4/5)下的mask图
        在level0提取annotation，然后按照像素画好，之后由fill hole补充。
    output：binary mask
'''
# 线程日志
log = util.Log('./log_wsi_otsu.txt')


def draw_mask(image, xml_path, scale=1, offset=(0, 0), width=10, cycle=False):
    """
        在image上画出标注信息
    """
    if not os.path.exists(xml_path):
        log('no annotation', xml_path)
        return
    annotation_polygon_list = util.get_scale_annotation(xml_path, scale, offset)

    # 在图上添加标注
    image_draw = ImageDraw.Draw(image)
    for annotation_polygon in annotation_polygon_list:
        draw_points = util.transfer_points_to_draw(annotation_polygon)
        if cycle:
            draw_points.append(draw_points[0])
            draw_points.append(draw_points[1])
        image_draw.line(tuple(draw_points), fill=(255, 255, 255, 128), width=width)


def main_flow(slide_path, xml_path, save_folder, down_sample):
    """
    每次切3200*3200block，缩小down_sample倍后，合并到 pseudo image 100*100
    :param slide_path:
    :param xml_path:保存32倍缩放下的原图、带标注原图、mask的前景、背景
    :param save_folder:
    :return:
    """
    save_basename = os.path.basename(slide_path).split('.')[0]
    slide = openslide.open_slide(slide_path)
    if os.path.exists('%s/%s_resize_%02d_mask' % (save_folder, save_basename, down_sample)):
        return None
    scale_dim = [int(dim / down_sample) for dim in slide.level_dimensions[0]]
    # 图像的dim顺序和np的不同，需要注意下转换
    pseudo_np = np.zeros([scale_dim[1], scale_dim[0], 3])

    # 生成前景背景
    pseudo_image = Image.fromarray(np.uint8(pseudo_np))

    # 保存level0缩小的原图、及level0缩小的标注图
    draw_mask(pseudo_image, xml_path, scale=down_sample, width=1, cycle=True)
    mask_np = np.asarray(pseudo_image)
    distance = nd.distance_transform_edt(mask_np[:, :, 0])
    fill_hole = nd.morphology.binary_fill_holes(distance)

    # 保存mask轮廓图片
    pseudo_image.save('%s/%s_resize_%02d_mask.png' % (save_folder, save_basename, down_sample))

    # 保存3通道mask可视化图片
    pseudo_np[:, :, 0][np.where(fill_hole > 0)] = 255
    pseudo_np[:, :, 1][np.where(fill_hole > 0)] = 255
    pseudo_np[:, :, 2][np.where(fill_hole > 0)] = 255
    pseudo_image = Image.fromarray(np.uint8(pseudo_np))
    pseudo_image.save('%s/%s_resize_%02d_mask_1.png' % (save_folder, save_basename, down_sample))

    # 保存单通道mask矩阵
    fill_hole = np.transpose(fill_hole)  # 将矩阵转置，变为与图像同样size
    np.save('%s/%s_resize_%02d_mask' % (save_folder, save_basename, down_sample), fill_hole)
    # 关闭线程和IO，避免内存问题
    slide.close()


# main arg parse
parser = argparse.ArgumentParser(description='用otsu方式阈值，取出wsi的前景区域')
parser.add_argument('-si', '--start_index', default=0, type=int, help='start index from glob.glob list')
parser.add_argument('-ei', '--end_index', default=0, type=int, help='end index from glob.glob list')
parser.add_argument('-ir', '--image_root',
                    default='/root/workspace/dataset/CAMELYON16/testing/images/',
                    type=str, help='待处理图片根目录')
parser.add_argument('-ar', '--annotation_root',
                    default='/root/workspace/dataset/CAMELYON16/testing/lesion_annotations',
                    type=str, help='待处理图片对应lesion annotation位置')
parser.add_argument('-sf', '--save_folder',
                    default='./wsi_otsu_save',
                    type=str, help='wsi 前、背景存储')
parser.add_argument('-ds', '--down_sample', default=32, type=int, help='缩放的倍数，level 5 的倍数约为1/32')

if __name__ == '__main__':
    args = parser.parse_args()

    range_start = args.start_index
    image_root = args.image_root
    annotation_root = args.annotation_root
    save_folder = args.save_folder
    down_sample = args.down_sample

    tif_list = glob.glob(os.path.join(image_root, '*.tif'))
    tif_list.sort()

    range_end = args.end_index if args.end_index != 0 else len(tif_list)
    log('start from index:%d' % range_start, 'image_root:%s' % image_root, 'annotation_root:%s' % annotation_root,
        'tif_list:%d' % len(tif_list))
    for i in range(range_start, range_end):
        tif_path = tif_list[i]
        tif_basename = os.path.basename(tif_path)
        log('----------- start process otsu %03d/%03d, %s ----------' % (i, range_end, tif_path))
        process_start = time.time()
        xml_path = os.path.join(annotation_root, '%s.%s' % (tif_basename.split('.')[0], 'xml'))
        main_flow(tif_path, xml_path, save_folder, down_sample)
        log('%03d/%03d, %s,time cost:%.4f' % (i, range_end, tif_basename, time.time() - process_start))
