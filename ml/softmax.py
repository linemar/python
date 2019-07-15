'''
@Descripttion: softmax程序实现
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-07-15 16:49:38
'''

import numpy as np

def autograd():
    pass

def softmax():
    pass

#s损失函数模型
def lose_func():
    pass 


#初始化参数模型, 根据输入数据与输出输出数据，确定w与b
def initial_model(num_inputs, num_outputs):
    
    w = np.random.nomal(scale = 0.01, shape = (num_inputs, num_outputs))
    b = np.zeros(num_outputs)

    return w, b

def data():
    pass