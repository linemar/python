'''
@Descripttion: 在单层softmax函数的基础上增加relu层
@version: 
@Author: SunZewen
@Date: 2019-08-16 09:47:16
@LastEditors: SunZewen
@LastEditTime: 2019-08-16 17:21:11
'''

from fashion_mnist import extract_train_img_data 
from fashion_mnist import extract_train_label_data 
from fashion_mnist import load_data 
# from fashion_mnist import show_image

import numpy as np
from math import *
import os
import time
import matplotlib 
import gzip
from matplotlib.font_manager import FontProperties 

np.random.seed(1)

#将输入数据进行处理，防止求softmax时值太大溢出
def initial_input(input):
    input_new = input / 255
    
    return input_new

def to_ont_hot(Y):
    new_Y = np.zeros(shape = (60000, 10))

    for i in range(len(Y)):
        new_Y[i][Y[i]] = 1

    return new_Y

#初始化模型
def initial_model(inputDim, outputDim):
    weights = np.random.rand(inputDim, outputDim)
    biases = np.random.rand(1, outputDim)

    return weights, biases

def output(W, b, x):
    
    o= x.dot(W) + b
    
    return o

def softmax(o):
    
    exp_o = np.exp(o)
    sum = np.sum(exp_o)

    return exp_o / sum


def softmax_loss(Y, Y_hat):

    loss = Y_hat - Y

    return loss


def softmaxGD(loss, X):
    
    # grad_W = loss.reshape(10, 1).dot(X)
    grad_W = X.reshape(784, 1).dot(loss)
    grad_b = loss

    return grad_W, grad_b

def update_para(W, b, grad_W, grad_b, lr):
    W -= grad_W * lr 
    b -= grad_b * lr

    return W, b

def error(Y, Y_hat):

    Y_hat_log = np.log(Y_hat)

    error = np.sum(-1 *  Y * Y_hat_log)

    return error

def accuracy(W, b, X, labels):
    
    out_list = []
    for i in range(len(X)):
        out = output(W, b, X[i])
        y_hat = softmax(out)
        out_list.append(y_hat.argmax())

    count = 0

    for  i in range(len(X)):
        if labels[i] == out_list[i]:
            count += 1
    
    return count/len(X)
    # print('rate : %f' %(count/len(X)))


def train(W, b, X, Y, lr, num_epoch):

    for j in range(num_epoch):
        print('epoch : %d' %(j))

        grad_W = np.zeros(shape = (784, 10))
        grad_b = np.zeros(shape = (1, 10))
        e = 0.
        
        for i in range(len(X)):
            x = X[i].reshape(1, 784)
            y = Y[i].reshape(1, 10)

            o = output(W, b, x)
            y_hat = softmax(o)

            loss = softmax_loss(y, y_hat)
            g_W , g_b = softmaxGD(loss, x)
            grad_W += g_W
            grad_b += g_b

            e += error(y, y_hat)

        grad_W = grad_W / len(X)
        grad_b = grad_b / len(X)

        W, b = update_para(W, b, grad_W, grad_b, lr)

        rate = accuracy(W, b, X, labels)

        print('error : %f' %(e/len(X)))
        print('accuracy rate : %f' %(rate))
        

    
    return W, b



if __name__ == "__main__":


    font = FontProperties(fname = r"/mnt/c/Windows/Fonts/simsun.ttc", size = 6)

    data_path = '../../data/fashion'
    class_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋','包', '短靴']

    file_list = ['t10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz',
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz']

    for i in file_list:
        load_data(data_path, i)

    headers, images = extract_train_img_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))

    # print(image)
    # print(type(images))

    header_array = np.frombuffer(headers, dtype = '>u4')
    print(header_array)

    X = np.zeros(shape = (60000, 784))

    for i in range(header_array[1]):
        X[i] = np.frombuffer(images[i], dtype = '>u1') / 255

    # print(img_array[0])
    # print(len(img_array[0]))

    labels = extract_train_label_data(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))

    Y = to_ont_hot(labels)

    #初始化模型，输入参数
    W, b = initial_model(28 * 28, 10)
    
    uW, ub = train(W, b, X, Y, 0.05, 100)

    # print(labels[0:200])

    test_headers, test_images = extract_train_img_data(os.path.join(data_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = extract_train_label_data(os.path.join(data_path, 't10k-labels-idx1-ubyte.gz'))

    print(test_headers)

    test_header_array = np.frombuffer(test_headers, dtype = '>u4')
    
    test_X = np.zeros(shape = (10000, 784))

    for i in range(test_header_array[1]):
        test_X[i] = np.frombuffer(test_images[i], dtype = '>u1') / 255

    rate = accuracy(uW, ub, test_X, test_labels)
    print('accuracy rate : %f' %(rate))
    
