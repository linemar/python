'''
@Descripttion: softmax 简单版本程序实现，生成简单的模型与数据，去查找识别fashion_mnist的模型哪里有问题
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-08-15 16:55:36
'''

import numpy as np
from math import *
import os
import time
import matplotlib 
import gzip
from matplotlib.font_manager import FontProperties 

debug = 0

#计算预测值
def output(W, b, X):

    O = X * W + b
    Y_hat = softmax(O)

    return Y_hat


#softmax函数
def softmax(O):

     O = O.reshape(1,3)
     max = np.max(O, axis = 1)

     sum = np.exp(O - max).sum(axis = 1)
     
     return np.exp(O - max)/sum

#s损失函数模型
def crossentropy(y, y_hat):
     
     # for i in range(len(y)):
     #      loss += y[i] * log(y_hat[i]) 

     return loss

#计算反向传播的梯度
#此处应该计算的小批量的平均梯度
def grad(y, y_hat, x):
     
     grad_w = (y_hat - y) * x
     grad_b = (y_hat - y)
     
     return grad_w, grad_b


#初始化参数模型, 根据输入数据与输出输出数据，确定w与b
def initial_model(num_inputs, num_outputs):
    
#     W = np.zeros(shape = (num_inputs, num_outputs))
     W = np.random.randint(1,5, [num_inputs, num_outputs])
     b = np.zeros(shape = (1, num_outputs))

     return W, b


#损失函数
def loss(W , b , images, labels):

     loss = 0.

     Y = np.zeros(shape = (len(images), 10), dtype=np.int32)

     for i in range(len(images)):
          Y[i][labels[i]] = 1

     O = np.zeros(shape = (len(images), 10), dtype=np.float64)
     Y_hat = np.zeros(shape = (len(images), 10), dtype=np.float64)
     
     for j in range(len(images)):
          image = images[j]
          O[j] = image.dot(W) + b
          Y_hat[j] = softmax(O[j])

          for i in range(10):
               if Y[j][i] != 0 and Y_hat[j][i] == 0 :
                    loss += 1.0

     return loss/len(images)


#用导数的定义去求导,判断是否是导数出错
def grad_def(W, b, image, label):

     Y = np.zeros(shape = (1, 10), dtype=np.float64)
     O = np.zeros(shape = (1, 10), dtype=np.float64)
     Y_hat = np.zeros(shape = (1, 10), dtype=np.float64)
     grad_W = np.zeros(shape = (len(W), 10), dtype=np.float64) 

     O = image.dot(W) + b
     Y_hat = softmax(O)
     
     tmp_Y_hat = np.zeros(shape = (1, 10), dtype=np.float64)
     tmp_O = np.zeros(shape = (1, 10), dtype=np.float64)
     tmp_W = W
     h = 0.
     tmp_h = 0.

     for i in range(784):
          for j in range(10):
               tmp_W = tmp_W[i][j] + 0.00001
               tmp_O = image.dot(tmp_W) + b
               tmp_Y_hat = softmax(tmp_O)

               for k in range(10):
                    h += - Y[k] * log(Y_hat[k])
                    tmp_h += - Y[k] * log(tmp_Y_hat[k])

               grad_W[i][j] = (h - tmp_h)/0.00001


     return grad_W
     
#训练函数，模型（W,b)，数据集（图片，标签, 学习率，梯度，误差范围
def train(W, b, train_data, labels, lr, num_epoch):
     print('start training ...')

     Y = Y_hat = np.zeros(shape = (len(train_data), 3), dtype=np.float64)
     for i in range(len(labels)):
          Y[i][labels[i]] = 1; 

     #总迭代次数
     for j in range(num_epoch):
          print('epoch : %d' %(j))

          #每次大循环开始，初始化 O, grad_W, grad_b
          # O 二维数组，存储中间计算结果, 
          # grad_W，矩阵，二维数组，所有样本的梯度和
          # grad_b，向量，一维数组，所有样本的梯度和
          O = np.zeros(shape = (1, 3), dtype=np.float64)
          Y_hat = np.zeros(shape = (1, 3), dtype=np.float64)
          grad_W = np.zeros(shape = (5, 3), dtype=np.float64)  
          grad_b = np.zeros(shape = (1, 3), dtype=np.float64)

          for j in range(len(train_data)):
               #根据初始条件计算预测值
               # images[j].dot(W) 
               O = train_data[j].dot(W) + b
               Y_hat = softmax(O)
               
               if debug == 1: 
                    print('X[%d]' %(j))
                    print(X[j])

                    print('Y[%d]' %(j))
                    print(Y[j])
                    print('')

                    print('W')
                    print(W)

                    print('b')
                    print(b)
                    print('')

                    print('O')
                    print(O)
                    
                    print('Y_hat')
                    print(Y_hat)

                    print('')

                    
                    print('dw = ')
                    print(np.dot(train_data[j].reshape(5, 1) ,(Y_hat - Y[j]).reshape(1, 3)))

                    print('db = ')
                    print(Y_hat - Y[j])

                    print('')

               #计算全局偏导
               #偏导计算公式为(dw = (y - y_hat)*x), x即输入images
               grad_W += np.dot(train_data[j].reshape(5, 1) ,(Y_hat - Y[j]).reshape(1, 3))
               grad_b += Y_hat - Y[j]

          #更新权重参数
          W = W - lr * grad_W / len(train_data)
          b = b - lr * grad_b / len(train_data)
          
          Y1 = test(W, b, train_data)
          rate = accuracy(labels, np.nonzero(Y1)[1])

          print('rate : %f' %(rate))

          if debug == 1: 
               print('W')
               print(W)
               
               print('b')
               print(b)

               print('')

     return W, b


#测试函数，使用测试数据集进行测试
def test(W, b, data_set):

     O = np.zeros(shape = (1, 3), dtype=np.float64)
     Y_hat = np.zeros(shape = (len(data_set), 3), dtype=np.float64)
     
     for j in range(len(data_set)):
          O = data_set[j].dot(W) + b
          Y_hat[j] = softmax(O)

     return Y_hat


#统计准确率
def accuracy(labels, Y_hat):
     count = 0
     for i in range(len(labels)):
          if labels[i] == Y_hat[i]:
               count += 1

     return count/len(labels)

def genarate_train_data(num):

     X = np.random.randint(0, 300, [num, 5])
     Y = X.sum(axis = 1)

     for i in range(num):
          if Y[i] < 500:
               Y[i] = 0
          
          if Y[i] >= 500 and Y[i] < 1000 :
               Y[i] = 1
          
          if Y[i] >= 1000 :
               Y[i] = 2
     
     return X, Y


#主函数
if __name__ == "__main__":
    
     #生成训练数据
     X, Y = genarate_train_data(1000)

     if debug == 1: 
          print('X')
          print(X)
          print('Y')
          print(Y)
     
     #初始化模型，输入参数
     W, b = initial_model(5, 3)
     print('initial_model : ')
     
     if debug == 1: 
          print('W')
          print(W)

          print('b')
          print(b)

     print('train ')

     W, b = train(W, b, X, Y, 0.1, 300)
     Y1 = test(W, b, X[:10])

     # print(Y[0:10])
     # print(np.nonzero(Y1[0:10])[1])