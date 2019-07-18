'''
@Descripttion: softmax程序实现
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-07-18 22:15:04
'''

import numpy as np

def autograd():
    pass

#将输入数值转化为概率分布
def initial_input(vector):

    # vector.sum(axis = 1)
    print('np.exp(vector) : ')
    print(np.exp(vector))
    # print(vector.sum(axis = 0))

    print('np.exp(vector).sum(axis = 0) : ')
    print(np.exp(vector).sum(axis = 0))

    # new_vector = np.exp(vector) / vector.sum(axis = 0)
    new_vector = np.exp(vector) / np.exp(vector).sum(axis = 0)
    return new_vector
    

#计算预测值
def comput_y_hat(W, X):

    result = X * W

    return result


def softmax(O):
     Y = []
     sum = np.exp(O).sum
     for i in range(0, len(Y)):
          Y.append(O[i]/sum)

     return Y


#s损失函数模型
def lose_func(y, y_hat):
     
     loss = 0
     
     for i in range(0, len(y)):
          loss += y[i] * np.log(y_hat[i])

     return loss


#初始化参数模型, 根据输入数据与输出输出数据，确定w与b
def initial_model(num_inputs, num_outputs):
    
    w = np.random.normal(scale = 0.01, size = (num_inputs, num_outputs))
    b = np.zeros(shape = (1, 10))

    return w, b

def data():
    pass


if __name__ == "__main__":
    
    #初始化模型，输入参数
    W, b = initial_model(28 * 28, 10)

    print('w.shape : ' + str(w.shape))
    # print(w.shape)
    print(w)

    print('b.shape : '+ str(b.shape))
    print(b)


    #处理输入数据
    new_w = initial_input(w)
    print(new_w)

