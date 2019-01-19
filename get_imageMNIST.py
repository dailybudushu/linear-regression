# -*- coding: utf-8 -*-
# @File  : get_imageMNIST.py
# @Author:
# @Date  : 2018/10/7
# @Desc  : to get image from MNIST .Copy from https://www.jianshu.com/p/84f72791806f
import matplotlib.pyplot as plt
import os
import struct
import numpy as np
from numpy import *
import operator
import os
import numpy as np
import time
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib import  cm
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import struct
import math


def decode_idx3_ubyte(idx3_ubyte_file, num_images=60000):
    """
    解析idx3文件的通用函数
    :param num_images:
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    # 其中>为大端(是指数据的低位保存在内存的高地址中，而数据的高位，保存在内存的低地址中),i为无符号整数.
    fmt_header = '>iiii'
    magic_number, total_num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, total_num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # 这里的B为bit，这段代码的意思是读取image_size个bit，每个bit是一个像素点
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    print("共提取" + str(num_images) + "张图片")
    return images


def decode_idx1_ubyte(idx1_ubyte_file, num_images=60000):
    """
    解析idx1文件的通用函数
    :param num_images:
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, total_num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, total_num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    print("共提取"+str(num_images)+"个标签")
    return labels


def load_train_images(idx_ubyte_file, num_images=60000):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param num_images:
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, num_images)


def load_train_labels(idx_ubyte_file, num_images=60000):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param num_images:
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, num_images)


def load_test_images(idx_ubyte_file, num_images=10000):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param num_images:
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, num_images)


def load_test_labels(idx_ubyte_file, num_images=10000):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param num_images:
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, num_images)


def load_test_labels_one_hot(idx_ubyte_file, num_images=10000):
    """
       解析idx1文件的通用函数
       :param num_images:
       :param idx_ubyte_file: idx1文件路径
       :return: 数据集
       """
    # 读取二进制数据
    bin_data = open(idx_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, total_num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, total_num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.zeros((num_images, 10), int)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        number = struct.unpack_from(fmt_image, bin_data, offset)[0]
        labels[i][number] = 1
        offset += struct.calcsize(fmt_image)
    print("共提取" + str(num_images) + "个标签")
    return labels


def load_train_labels_one_hot(idx_ubyte_file, num_images=60000):
    return load_test_labels_one_hot(idx_ubyte_file, num_images)


def normalization(images):
    img_normlization = np.round(images / 255)
    return img_normlization


