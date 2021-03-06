'''
@Descripttion: 在单层softmax函数的基础上增加relu层
@version: 
@Author: SunZewen
@Date: 2019-08-16 09:47:16
@LastEditors: SunZewen
<<<<<<< HEAD
@LastEditTime: 2019-08-21 08:23:46
=======
@LastEditTime: 2019-08-21 12:57:35
>>>>>>> bfbbbe3185a80536d8393df03358c1d3f8f1b5fb
'''

from fashion_mnist import extract_train_img_data 
from fashion_mnist import extract_train_label_data 
from fashion_mnist import load_data 
# from fashion_mnist import show_image

import numpy as np
from math import *
import random
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
def initial_model(inputDim, outputDim, numberOfLayers):

    weights = [None for i in range(numberOfLayers)]
    biases = [None for i in range(numberOfLayers)]
    
    for i in range(numberOfLayers - 1):
        weights[i] = np.random.rand(inputDim, inputDim)
        biases[i] = np.random.rand(1, inputDim)
    
    weights[numberOfLayers - 1] = np.random.rand(inputDim, outputDim)
    biases[numberOfLayers - 1] = np.random.rand(1, outputDim)

    return weights, biases

#定义每层的输出函数   
def output(W, b, x, numberOfLayers):

    h = []
    ho = []

    ho.append(x)

    for i in range(0, numberOfLayers - 1):
        h.append((ho[i].dot(W[i]) + b[i]) / 784)   #计算第i层的输出, 将输入都缩小至0 - 1，避免softmax溢出
        ho.append(ReLU(h[i]))          #将其转换为ReLU
    
    h.append(ho[numberOfLayers - 1].dot(W[numberOfLayers - 1]) + b[numberOfLayers - 1])   #计算最后一层的输出
    a = softmax(h[numberOfLayers - 1])

    return h, ho, a   #返回每一层的输出，与relu操作后的结果，并且计算最后的预测值

def softmax(o):

    # print(o)
    exp_o = np.exp(o)
    sum = np.sum(exp_o)

    return exp_o / sum

def ReLU(input):
    return np.maximum(input, 0)
    
def ReLUGD(input):
    output  = np.maximum(input, 0)  #先将数组中的负数归零
    output = np.minimum(output, 1) #将大于1的置1
    output = np.maximum(output, 1) #将大于0, 小于1的置1

    return output

def softmax_loss(Y, Y_hat):

    loss = Y_hat - Y

    return loss


def softmaxGD(loss, X):
    
    # grad_W = loss.reshape(10, 1).dot(X)
    grad_W = X.reshape(784, 1).dot(loss)
    grad_b = loss

    return grad_W, grad_b

'''
@name:
@test: test font
@msg: 计算每一层的梯度，并将其返回
@param {type}
@return: grad_W, grad_b
'''
def grad(W, b, x, y, numberOfLayers):
    h, ho, a = output(W, b, x, numberOfLayers)

    grad_W = [0 for i in range(numberOfLayers)]
    grad_b = [0 for i in range(numberOfLayers)]

    for i in range(numberOfLayers):
        # print('i : %d' %(i))
        grad_W[numberOfLayers - 1 - i] = ho[numberOfLayers - 1 - i].reshape(784, 1).dot((a - y)) #将x当做ho0
        grad_b[numberOfLayers - 1 - i] = a - y
        
        if i >= 1 :

            grad_W[numberOfLayers - 1 - i] = np.multiply(np.dot(grad_W[numberOfLayers - 1 - i], W[numberOfLayers -1].reshape(10, 784)), np.tile(ReLUGD(h[numberOfLayers -2]).reshape(1, 784), (784, 1)))
            grad_b[numberOfLayers - 1 - i] = np.multiply(np.dot(grad_b[numberOfLayers - 1 - i], W[numberOfLayers -1].reshape(10, 784)), ReLUGD(h[numberOfLayers -2]).reshape(1, 784))

        
        for j in range(2, i):
            
            grad_W[numberOfLayers - 1 - i] = np.multiply(np.dot(grad_W[numberOfLayers - 1 - i], W[numberOfLayers - 1 - j]), np.tile(ReLUGD(h[numberOfLayers -1 - j]).reshape(1, 784), (784, 1)))
            grad_b[numberOfLayers - 1 - i] = np.multiply(np.dot(grad_b[numberOfLayers - 1 - i], W[numberOfLayers - 1 - j]), ReLUGD(h[numberOfLayers -1 - j]).reshape(1, 784))

    return grad_W, grad_b


def update_para(W, b, grad_W, grad_b, numberOfLayers, lr):
    for i in range(numberOfLayers):
        W[i] -= grad_W[i] * lr 
        b[i] -= grad_b[i] * lr

    return W, b

def error(Y, Y_hat):

    Y_hat_log = np.log(Y_hat)

    error = np.sum(-1 *  Y * Y_hat_log)

    return error

def accuracy(W, b, X, Y, numberOfLayers):

    out_list = []
    for i in range(len(X)):
        h, ho, out = output(W, b, X[i], numberOfLayers)
        # y_hat = softmax(out)
        out_list.append(out.argmax())

    count = 0

    for  i in range(len(X)):
        if Y[i] == out_list[i]:
            count += 1
    
    return count/len(X)
    # print('rate : %f' %(count/len(X)))


#使用小批量梯度下降法进行训练
def batch_size_train(W, b, X, Y, numberOfLayers, lr, batch_size, num_epoch):

    order = np.arange(0, 60000, dtype=np.int32)

    for j in range(num_epoch):
        print('epoch : %d' %(j))
  
        e = 0.

        grad_W = [0 for i in range(numberOfLayers)]
        grad_b = [0 for i in range(numberOfLayers)]

        # for i in range(numberOfLayers - 1):
        #     grad_W.append(np.zeros(shape = (784, 784)))
        #     grad_b.append(np.zeros(shape = (1, 784)))

        # grad_W.append(np.zeros(shape = (784, 10)))
        # grad_b.append(np.zeros(shape = (1, 10)))
        A = 0.

        for i in range(int(len(X)/batch_size)):
            # print('i : %d' %(i))
               
            random.shuffle(order)

            for k in order[0 : batch_size]:
                x = X[k].reshape(1, 784)
                y = Y[k].reshape(1, 10)

                h, ho, A = output(W, b, x, numberOfLayers)
                g_W , g_b = grad(W, b, x, y, numberOfLayers)

                for j in range(numberOfLayers):

                    grad_W[j] += g_W[j]
                    grad_b[j] += g_b[j]
          
                e += error(y, A)
            
            for m in range(numberOfLayers):
                grad_W[m] = grad_W[m] / batch_size
                grad_b[m] = grad_b[m] / batch_size
    
            W, b = update_para(W, b, grad_W, grad_b, numberOfLayers, lr)
  
        rate = accuracy(W, b, X, labels, numberOfLayers)
  
        print('error : %f' %( e / len(X) ))
        print('accuracy rate : %f' %( rate ) )

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

    #训练数据
    headers, images = extract_train_img_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
    header_array = np.frombuffer(headers, dtype = '>u4')
    print(header_array)

    X = np.zeros(shape = (60000, 784))
    for i in range(header_array[1]):
        X[i] = np.frombuffer(images[i], dtype = '>u1') / 255

    labels = extract_train_label_data(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))


    #测试数据
    test_headers, test_images = extract_train_img_data(os.path.join(data_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = extract_train_label_data(os.path.join(data_path, 't10k-labels-idx1-ubyte.gz'))

    print(test_headers)
    test_header_array = np.frombuffer(test_headers, dtype = '>u4')
    
    test_X = np.zeros(shape = (10000, 784))

    for i in range(test_header_array[1]):
        test_X[i] = np.frombuffer(test_images[i], dtype = '>u1') / 255

    #将对应的标签转化为one-hot向量
    Y = to_ont_hot(labels)
        
    #初始化模型，输入参数
    W, b = initial_model(28 * 28, 10, 2)

    W, b = batch_size_train(W, b, X, Y, 2, 0.5, 200, 10)
    # train(W, b, X, Y, 0.05, 100)
    rate = accuracy(W, b, test_X, test_labels, 2)
    print('accuracy rate : %f' %(rate))
    
