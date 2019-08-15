'''
@Descripttion: softmax程序实现
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-08-15 23:00:33
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


#将输入数值转化为概率分布
# def initial_input(vector):

#     # vector.sum(axis = 1)
#     print('np.exp(vector) : ')
#     print(np.exp(vector))
#     # print(vector.sum(axis = 0))

#     print('np.exp(vector).sum(axis = 0) : ')
#     print(np.exp(vector).sum(axis = 0))

#     # new_vector = np.exp(vector) / vector.sum(axis = 0)
#     new_vector = np.exp(vector) / np.exp(vector).sum(axis = 0)
#     return new_vector

def initial_input(vector):

     for i in vector:
          i = i/256

     return vector
    

#计算预测值
def output(W, b, X):

    O = X * W + b
    Y_hat = softmax(O)

    return Y_hat


#softmax函数
def softmax(O):
     O = O.reshape(1, 10)

     max = np.max(O)

     # print('O')
     # print(O)
     # print('max')
     # print(max)

     sum = np.exp(O - max).sum(axis = 1)
     if sum > 100:
          print('sum')
          print(sum)
     # for i in range(0, len(y)):
     #      loss += np.exp(y[i])/sum
     
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
    
    W = np.random.normal(scale = 0.01, size = (num_inputs, num_outputs))
    b = np.zeros(shape = (1, num_outputs))

    return W, b

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

#训练函数，模型（W,b)，数据集（图片，标签, 学习率，梯度，误差范围
def train(W, b, images, labels, lr, num_epoch):
     

     Y = np.zeros(shape = (len(images), 10), dtype=np.int32)

     for i in range(len(images)):
          Y[i][labels[i]] = 1

     print(Y[0:10,])

     #总迭代次数
     for j in range(num_epoch):
          print('epoch : %d' %(j))

          #每次大循环开始，初始化 O, grad_W, grad_b
          # O 二维数组，存储中间计算结果, 
          # grad_W，矩阵，二维数组，所有样本的梯度和
          # grad_b，向量，一维数组，所有样本的梯度和
          O = np.zeros(shape = (1, 10), dtype=np.float64)
          Y_hat = np.zeros(shape = (1, 10), dtype=np.float64)
          grad_W = np.zeros(shape = (len(W), 10), dtype=np.float64)
          grad_b = np.zeros(shape = (1, 10), dtype=np.float64)
          
          for j in range(len(images)):
               image = images[j].reshape(1, 784)
               #根据初始条件计算预测值
               # images[j].dot(W) 
               # O = image.dot(W.reshape(784, 10)) + b
               O = image * W + b
               Y_hat = softmax(O)
          
               # print('O')
               # print(O[0])
               # print('Y_hat')
               # print(Y_hat[0])
               # print(Y_hat[0].shape)
               # re_y = Y_hat[0].reshape(10, 1)
               # print(re_y)
               # print(re_y.shape)
          
          # 根据图片数量进行训练
          # for j in range(len(images)):

               # 计算全局偏导
               # 偏导计算公式为(dw = (y - y_hat)*x), x即输入images
               grad_W = grad_W + np.dot(image.reshape(784, 1) ,(Y_hat.reshape(1, 10) - Y[j].reshape(1, 10)).reshape(1, 10))
               grad_b = grad_b + Y_hat.reshape(1, 10) - Y[j].reshape(1, 10)

          # print('grad_W')
          # print(grad_W)

          # print('grad_W / len(images)')
          # print(grad_W/ len(images))

          #更新权重参数
          W = W - lr * grad_W / len(images)
          b = b - lr * grad_b / len(images)

          # print('W')
          # print(W)
          # print('b')
          # print(b)

          OY = test(W, b, images)

          #Y是一个二维数组，按照行取出其行中不为零的列标，方便与标准值进行比较
          out_Y = np.nonzero(OY)[1]

          rate = accuracy(labels, out_Y)
          print('accuracy : %f' %(rate))

          # e = loss(W , b , images, labels)
          # print('loss : %f'%(e))

     return W, b


#测试函数，使用测试数据集进行测试
def test(W, b, images):

     O = np.zeros(shape = (len(images), 10), dtype=np.float64)
     Y_hat = np.zeros(shape = (len(images), 10), dtype=np.float64)
     
     for j in range(len(images)):
          image = images[j]
          O[j] = image.dot(W) + b
          Y_hat[j] = softmax(O[j])

     return Y_hat


#统计准确率
def accuracy(labels, Y_hat):
     count = 0
     for i in range(len(labels)):
          if labels[i] == Y_hat[i]:
               count += 1

     return count/len(labels)


def show_image(class_names, labels, O, images):
     plt.figure()
 
     for i in range(1,33):
         plt.subplot(4,8,i)
 
         plt.xticks([])
         plt.yticks([])
 
         plt.imshow(images[i - 1].reshape(28, 28), cmap=plt.cm.binary)
         plt.xlabel(class_names[labels[i - 1]], fontproperties=font)
         plt.xlabel(class_names[O[i - 1]], fontproperties=font)
 
     plt.show()


#主函数
if __name__ == "__main__":
    
     #初始化模型，输入参数
     W, b = initial_model(28 * 28, 10)

     # print('w.shape : ' + str(W.shape))
     # # print(w.shape)
     # print(W)

     # print('b.shape : '+ str(b.shape))
     # print(b)


     #处理输入数据
     # new_W = initial_input(W)
     # print(new_W)

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
     
     img_array = []
     for i in range(header_array[1]):
          img_array.append(np.frombuffer(images[i], dtype = '>u1'))
     
     # print(img_array[0])
     # print(len(img_array[0]))

     labels = extract_train_label_data(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))

     W, b = train(W, b, img_array, labels, 0.01, 50)
     Y = test(W, b, img_array)

     # print(labels)
     # print(np.nonzero(Y)[1])
     # print(Y.shape)
     
     #Y是一个二维数组，按照行取出其行中不为零的列标，方便与标准值进行比较
     out_Y = np.nonzero(Y)[1]

     #计算准确率，并输出
     rate = accuracy(labels, out_Y)
     print('accuracy : %f' %(rate))
     