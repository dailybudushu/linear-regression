# -*- coding: utf-8 -*-
# @File  : mnist_linear_classification.py
# @Author: LambdaX
# @Date  : 2018/11/22
# @Desc  : mnist set linear classification

import get_imageMNIST
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def softmax(labels):
    result = np.exp(labels)/np.sum(np.exp(labels))
    return result


def calculate_p(w, x, b):
    result = np.dot(w, x)+b
    result = softmax(result)
    return result


def calculate_loss(w, x, b, y):
    log_p = np.log(calculate_p(w, x, b))
    result = np.sum(np.dot(y, log_p))
    return -result


def calculate_derivative_b(w, x, b, y):
    result = calculate_p(w, x, b)-np.reshape(y, (10, 1))
    return result


def calculate_derivative_w(w, x, b, y):
    p = calculate_p(w, x, b)
    y = np.reshape(y, (10, 1))
    result = np.dot(p-y, np.reshape(x, (1, 784)))+0.001/784*w
    return result


def get_softmax_w_b(mnist_file_path, num=60000):
    train_images_path = mnist_file_path+'\\train-images.idx3-ubyte'
    train_labels_path = mnist_file_path+'\\train-labels.idx1-ubyte'
    test_images_path = mnist_file_path+'\\t10k-images.idx3-ubyte'
    test_labels_path = mnist_file_path+'\\t10k-labels.idx1-ubyte'
    train_images = get_imageMNIST.load_train_images(train_images_path, num)
    train_labels = get_imageMNIST.load_train_labels_one_hot(train_labels_path,num)
    test_images = get_imageMNIST.load_test_images(test_images_path)
    test_labels = get_imageMNIST.load_test_labels_one_hot(test_labels_path)
    train_images = get_imageMNIST.normalization(train_images)
    train_images = train_images.reshape((num, 784, 1))
    test_images = get_imageMNIST.normalization(test_images)
    test_images = test_images.reshape((10000, 784, 1))
    w = np.random.rand(10, 784) * 0.01
    b = np.random.rand(10, 1)*0.01
    sum_derivative_w = 0
    sum_derivative_b = 0
    step = 0.001
    loss = []
    for i in range(num):
        if i % 10 != 0:
            sum_derivative_w = sum_derivative_w+calculate_derivative_w(w, train_images[i], b, train_labels[i])
            sum_derivative_b = sum_derivative_b+calculate_derivative_b(w, train_images[i], b, train_labels[i])
        if i % 10 == 0:
            sum_derivative_w = sum_derivative_w+calculate_derivative_w(w, train_images[i], b, train_labels[i])
            sum_derivative_b = sum_derivative_b+calculate_derivative_b(w, train_images[i], b, train_labels[i])
            w = w-step*sum_derivative_w
            b = b-step*sum_derivative_b
            sum_derivative_b = 0
            sum_derivative_w = 0
    #     loss.append(calculate_loss(w, train_images[i], b, train_labels[i]))
    # x = range(num)
    # plt.plot(x, loss)
    return (w, b)


def get_accuracy(w, b, mnist_file_path):
    test_images_path = mnist_file_path + '\\t10k-images.idx3-ubyte'
    test_labels_path = mnist_file_path + '\\t10k-labels.idx1-ubyte'
    test_images = get_imageMNIST.load_test_images(test_images_path)
    test_labels = get_imageMNIST.load_test_labels_one_hot(test_labels_path)
    test_images = get_imageMNIST.normalization(test_images)
    test_images = test_images.reshape((10000, 784, 1))
    predict = []
    for i in range(10000):
        p = calculate_p(w, test_images[i], b)
        max_index = np.argwhere(p == np.max(p))
        max_row = max_index[0][0]
        if test_labels[i][max_row] == 1:
            predict.append(1)
    print('正确率:'+str(np.sum(predict)/10000))


wb = get_softmax_w_b('C:\\Users\\Lenovo\\Desktop\\mnist')
get_accuracy(wb[0], wb[1], 'C:\\Users\\Lenovo\\Desktop\\mnist')
