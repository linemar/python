'''
@Descripttion: softmax程序实现
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-07-19 16:00:26
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
def output(W, X):

    O = X * W

    return O


#softmax函数
def softmax(y):
     sum = np.exp(y).sum(axis = 0)
     
     for i in range(0, len(y)):
          loss += np.exp(y[i])/sum


#s损失函数模型
def crossentropy(y, y_hat):
     
     for i in range(0, len(y))
          loss += y[i] * log(y_hat[i]) 

     return loss


#计算反向传播的梯度
def grad(y, y_hat):
     
     grad = y_hat - y
     
     return grad


#初始化参数模型, 根据输入数据与输出输出数据，确定w与b
def initial_model(num_inputs, num_outputs):
    
    w = np.random.normal(scale = 0.01, size = (num_inputs, num_outputs))
    b = np.zeros(shape = (1, 10))

    return w, b


#训练函数，模型（W,b)，数据集（图片，标签,, 学习率，梯度，误差范围
def train(net, images, labels, lr, grad, loss):
     
     
     #w = w + lr * grad
     #b = b + lr * grad

     pass


#测试函数，使用测试数据集进行测试
def test(images, labels):
     pass


#统计准确率
def accuracy():
     pass


#主函数
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

