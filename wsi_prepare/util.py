import numpy as np
import sys
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
import math
from skimage.measure import points_in_poly
import multiprocessing


# 简易日志记录
class Log(object):
    def __init__(self, file='./log_demo.txt', console=True):
        print('log file in : %s' % file)
        self.file = file
        self.console = console
        self.lock = multiprocessing.Lock()

    def __call__(self, *args):
        self.lock.acquire()
        if self.console:
            print(*args)
        f = open(self.file, 'a')
        for arg in args:
            f.write(str(arg))
            f.write(',')
        f.write('\n')
        f.close()
        self.lock.release()


def get_scale_annotation(xml_path, scale=1, offset=[0, 0], circle=False):
    """
    获取标注信息
    :param xml_path: level0层的标注
    :param scale: 坐标缩放倍数，1==level0，2==level1 ...
    :param offset: 需要在level0上的偏移，截取局部展示图时使用
    :param circle: 获取到的annotation，首尾不相连，如果用line画线，为了可视化的circle效果，选择True
    :return:[[N, 2], [N, 2]]
    """
    annotation_polygon_list = []
    root = ET.parse(xml_path).getroot()
    annotations = root.findall('./Annotations/Annotation')

    for annotation in annotations:
        coordinates = annotation.findall('./Coordinates/Coordinate')
        pylygon = []
        for coord in coordinates:
            pylygon.append([(float(coord.get('X')) - offset[0]) / scale,
                            (float(coord.get('Y')) - offset[1]) / scale])
        # 改用line画图，需要将起点重复一次到最后，画出polygon
        if circle:
            pylygon.append(pylygon[0])
        annotation_polygon_list.append(pylygon)
    return annotation_polygon_list


def transfer_points_to_draw(points):
    """
    转换points到draw的格式
    :param points: [N, 2], draw[x1,y1,x2,y2]
    :return:
    """
    draw_points = []
    for point in points:
        draw_points.append(point[0])
        draw_points.append(point[1])
    return draw_points


def polygon_zoom(_old_points, h):
    """
    多边形轮廓按比例缩放，用向量的方式添加外扩
    :param _old_points: [N, 2]
    :param h:外部扩展的距离
    :return: new_points [N, 2]
    """
    new_points = []
    old_points = np.array(_old_points)
    #     points_in_poly([point], annotation_polygon)[0]

    # 逐次遍历定点，每次处理相邻的两个点，从夹角中线扩展
    for i in range(len(old_points)):
        old_point = old_points[i]
        point_0 = old_points[(i - 1 + len(old_points)) % len(old_points)]
        point_1 = old_points[(i + 1) % len(old_points)]

        #         print(type(point_0))
        v1 = point_0 - old_point
        v2 = point_1 - old_point
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        s = v1 + v2
        #         print(old_point)
        if points_in_poly([old_point + 0.1 * s], old_points)[0]:
            s = -s
        #         print(old_point, s, points_in_poly([old_point + 0.1*s], old_points)[0])
        dot_val = np.dot(v1, v2)
        cos_v = dot_val / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #         print(cos_v)
        cos_v = 1 if cos_v > 1 else -1 if cos_v < -1 else cos_v

        theta = math.acos(min(1, cos_v)) / 2

        if math.sin(theta) < 0.001:
            print(cos_v, theta, math.sin(theta))
            theta = 0.01
        s_norm = (h / math.sin(theta))
        if np.linalg.norm(s) == 0:
            continue

        s = s / np.linalg.norm(s) * s_norm
        new_points.append(old_point + s)

    return new_points
