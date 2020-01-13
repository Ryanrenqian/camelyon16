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

import os
import multiprocessing
import threading
import glob
import argparse
import util

'''
    issue：解决wsi在不同level的坐标偏移
        在level0层滑窗取3200*3200block，resize下采样到1/32，拼接成无偏移的1/32倍率图。
    output：在1/32图上的otsu前景、背景、mask
'''
# 线程日志
log = util.Log('./log_wsi_otsu.txt')


def wsi_otsu(image):
    """
    input: image(PIL.Image)
    output:
        region_origin - (np.array,m*n*3), 原图数据，用于对比
        region_forward - (np.array,m*n*3), 分割的前景
        region_backward - (np.array,m*n*3), 分割后的背景
        tissue_mask - mask, m*n
        count_true, count_false - otsu阈值保留的有效面积比例
    阈值的来源：是level5全图预先计算的otsu优化值
    默认会占满所有cpu用于加速，同时运行的其他程序会受影响
    """
    region_origin = np.array(image)

    region_backward = region_origin.copy()
    region_forward = region_origin.copy()
    # 颜色空间变换
    img_RGB = np.transpose(region_origin[:, :, 0:3], axes=[1, 0, 2])
    img_HSV = rgb2hsv(img_RGB)
    # otsu阈值处理前背景提取
    # print(threshold_otsu(img_RGB[:, :, 0]), threshold_otsu(img_RGB[:, :, 1]), threshold_otsu(img_RGB[:, :, 2]))
    background_R = img_RGB[:, :, 0] > 203
    background_G = img_RGB[:, :, 1] > 191
    background_B = img_RGB[:, :, 2] > 201
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > 0.1113
    '''如果仅使用用threadshold，中间会有部份白色脂肪区域被隔离'''
    rgb_min = 50
    min_R = img_RGB[:, :, 0] > rgb_min
    min_G = img_RGB[:, :, 1] > rgb_min
    min_B = img_RGB[:, :, 2] > rgb_min
    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    count_true = len(tissue_mask[tissue_mask == True])
    count_false = len(tissue_mask[tissue_mask == False])
    region_forward[np.transpose(tissue_mask) == False] = 0
    region_backward[np.transpose(tissue_mask) == True] = 0
    # 原图，前景，背景，mask，有效像素点
    return region_origin, region_forward, region_backward, tissue_mask, count_true, count_false


def draw_annotation(image, xml_path, scale=1, offset=[0, 0], width=10):
    """
        在image上画出标注信息
    """
    if not os.path.exists(xml_path):
        log('no annotation', xml_path)
        return
    annotation_polygon_list = util.get_scale_annotation(xml_path, scale, offset)

    pseudo_image_draw = ImageDraw.Draw(image)
    for annotation_polygon in annotation_polygon_list:
        draw_points = util.transfer_points_to_draw(annotation_polygon)
        pseudo_image_draw.line(tuple(draw_points), fill=(0, 0, 255, 128), width=width)


def gen_block_image(slide, x, y, queue, share_dict, lock):
    block_size = share_dict['block_size']
    real_size = int(block_size / share_dict['down_sample'])

    block_image = slide.read_region((x, y), 0, [block_size, block_size]).convert('RGB')
    block_image = block_image.resize([real_size, real_size])
    block_image_np = np.array(block_image)
    queue.put((x, y, block_image_np))
    lock.acquire()
    share_dict['block_generated'] += 1
    lock.release()


def start_block_producer(slide, multiprocess_block_queue, share_dict, lock):
    """
    处理图像的切取、缩放与结果合并
    :param slide:
    :param multiprocess_block_queue:
    :param share_dict:
    :param lock:
    :return:
    """
    # 下部和右边的边缘部份不处理
    x = 0
    p_list = []
    block_size = share_dict['block_size']
    while x + block_size < slide.level_dimensions[0][0]:
        y = 0
        while y + block_size < slide.level_dimensions[0][1]:
            lock.acquire()
            share_dict['block_produced'] += 1
            lock.release()
            # st = time.time()
            # 如果消费的比较慢，暂停新进程的创建，不需要使用线程池控制数目
            if share_dict['block_produced'] - share_dict['block_consumed'] > 200:
                print('wait', share_dict)
                time.sleep(3)
            p = multiprocessing.Process(target=gen_block_image,
                                        args=(slide, x, y, multiprocess_block_queue, share_dict, lock))
            p.start()
            p_list.append(p)
            # ed = time.time()
            # log(ed-st, share_dict['block_process'], share_dict['block_gen'],
            # share_dict['block_consumed'], multiprocess_block_queue.qsize())
            y += block_size
        x += block_size
        if share_dict['block_produced'] > share_dict['max_count']:
            break

    for p in p_list:
        p.join()


def block_consumer(block_queue, pseudo_np, share_dict, lock):
    """
    消费队列中的数据，合并到结果
    :param block_queue:
    :param pseudo_np:
    :param share_dict:
    :param lock:
    :return:
    """
    while share_dict['flag']:
        _down_sample = share_dict['down_sample']
        scale_size = int(share_dict['block_size'] / _down_sample)

        if share_dict['consumer_status'] == True:
            if block_queue.qsize() > 0:
                x, y, block_image_np = block_queue.get()

                # 合并小block
                x_block = int(x / _down_sample)
                y_block = int(y / _down_sample)
                pseudo_np[y_block: y_block + scale_size, x_block: x_block + scale_size, :] = block_image_np
                lock.acquire()
                share_dict['block_consumed'] += 1
                lock.release()
            if share_dict['block_produced'] > 0 and share_dict['block_generated'] == share_dict['block_produced']:
                if share_dict['block_consumed'] == share_dict['block_generated']:
                    share_dict['flag'] = False
                    break
    log('stop consumer', share_dict)


def start_block_consumer(multiprocess_block_queue, pseudo_np, share_dict, lock):
    """
    启动合并进程结果的消费
    :param multiprocess_block_queue:
    :param pseudo_np: 待合并的图像矩阵
    :param share_dict:多进程交换数据
    :param lock:进程lock
    :return:
    """
    t = threading.Thread(target=block_consumer, args=(multiprocess_block_queue, pseudo_np, share_dict, lock))
    t.start()


def main_flow(slide_path, xml_path, save_folder, down_sample):
    """
    每次切3200*3200block，缩小32倍后，合并到 pseudo image 100*100
    :param slide_path:
    :param xml_path:保存32倍缩放下的原图、带标注原图、mask的前景、背景
    :param save_folder:
    :return:
    """
    save_basename = os.path.basename(slide_path).split('.')[0]
    if os.path.exists('%s/%s_resize_%02d_anno.png' % (save_folder, save_basename, down_sample)):
        return None
    time_flow_start = time.time()
    slide = openslide.open_slide(slide_path)
    scale_dim = [int(dim / down_sample) for dim in slide.level_dimensions[0]]
    # 图像的dim顺序和np的不同，需要注意下转换+++++++++++++++++++++++++++++++++++++++++++++++++
    pseudo_np = np.zeros([scale_dim[1], scale_dim[0], 3])

    multiprocess_block_queue = multiprocessing.Queue(2500)
    lock = multiprocessing.Lock()
    share_dict = multiprocessing.Manager().dict()
    share_dict['down_sample'] = down_sample
    share_dict['block_size'] = 3200  # 每次从slide中取3200处理，经测试，这个大小比较合适
    share_dict['flag'] = True
    share_dict['block_produced'] = 0
    share_dict['block_generated'] = 0
    share_dict['block_consumed'] = 0
    share_dict['consumer_status'] = True
    share_dict['max_count'] = 15000

    start_block_consumer(multiprocess_block_queue, pseudo_np, share_dict, lock)
    start_block_producer(slide, multiprocess_block_queue, share_dict, lock)
    time_process = time.time()

    # 生成前景背景
    pseudo_image = Image.fromarray(np.uint8(pseudo_np))
    region, region_forward, region_backward, tissue_mask, count_True, count_False = wsi_otsu(pseudo_image)
    forward_image = Image.fromarray(np.uint8(region_forward))
    backward_image = Image.fromarray(np.uint8(region_backward))
    np.save('%s/%s_resize_%02d' % (save_folder, save_basename, down_sample), tissue_mask)
    forward_image.save('%s/%s_resize_%02d_forward.png' % (save_folder, save_basename, down_sample))
    backward_image.save('%s/%s_resize_%02d_backward.png' % (save_folder, save_basename, down_sample))
    time_otsu = time.time()

    # 保存level0缩小的原图、及level0缩小的标注图
    pseudo_image.save('%s/%s_resize_%02d.png' % (save_folder, save_basename, down_sample))
    draw_annotation(pseudo_image, xml_path, scale=down_sample, width=5)
    pseudo_image.save('%s/%s_resize_%02d_anno.png' % (save_folder, save_basename, down_sample))
    time_save = time.time()

    # 关闭线程和IO，避免阻塞
    share_dict['flag'] = False
    slide.close()
    log('handle block:%.4f otsu:%.4f save:%.4f' %
        ((time_process - time_flow_start), (time_otsu - time_process), (time_save - time_otsu)), share_dict)


# --- main arg parse ---
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

# python wsi_otsu.py -sf ./wsi_otsu_save/testing
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
