'''
@Descripttion: 
@version: 
@Author: SunZewen
@Date: 2019-08-01 08:29:00
@LastEditors: SunZewen
@LastEditTime: 2019-08-01 16:41:13
'''
from IPython import display
from matplotlib import pyplot as plt
import random
import numpy as np

'''
@name: 定义模型
@test: test font
@msg: 
@param {w, W的维度} 
@return: W,b
'''
def net(w):
    W = np.zeros((1,w), dtype = np.float32)
    b = 0
    return W,b

'''
@name: 生成训练数据 
@test: test font
@msg: 
@param {type} 
@return: 
'''
def genarate_data(W, b):

    #生成输入，即x
    featurs = np.random.normal(0, 1, (1000,2))
    
    #计算真实的y
    labels = W[0][0]* featurs[:,0] + W[0][1] * featurs[:,1] + b

    #添加误差
    labels += np.random.normal(0, 0.001, labels.shape)

    #显示生成的数据
    # display.set_matplotlib_formats('svg')
    # plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);

    return featurs, labels

'''
@name: 定义损失函数
@test: test font
@msg: 
@param {type} 
@return: 
'''
def loss_function(Y_hat, Y, n):
    loss =  np.sum(np.square(Y_hat - Y)) / n
    return loss

'''
@name: 定义优化函数
@test: test font
@msg: 
@param {type} 
@return: 
'''
def sgd(W, b, lr, batch_size):
    
    W[0][0] = W[0][0] - lr * grad /batch_size
    W[0][1] = W[0][1] - lr * grad /batch_size

    b = b - lr * grad /batch_size
    
    pass

'''
@name: 从样本空间中取出单次训练需要的数据
@test: test font
@msg: 
@param {type} 
@return: 
'''
def data_iter(features, labels, batch_size):

    order = random.shuffle(len(features))

    return features.take(order[:10]), labels.take(order[:10])


'''
@name: 模型训练
@test: test font
@msg: 
@param {type} 
@return: 
'''
def train(W, b, lr, features, labels, batch_size, epoch):

    for num in range(0, epoch):
        
        loss = 0;

        grad_w1 = 0
        grad_w2 = 0
        grad_b = 0
        
        for i in range(0, 1000):

            grad_w1 += features[i][0] * (features[i][0] * W[0][0] + features[i][1] * W[0][1] + b - labels[i])
            grad_w2 += features[i][1] * (features[i][1] * W[0][0] + features[i][1] * W[0][1] + b - labels[i])
            grad_b  += features[i][0] * W[0][0] + features[i][1] * W[0][1] + b - labels[i]

        W[0][0] = W[0][0] - lr * grad_w1 / 1000
        W[0][1] = W[0][1] - lr * grad_w2 / 1000

        b = b - lr * grad_b / 1000

        for i in range(0 , 1000):
            loss += 0.5 * (features[i][0] * W[0][0] + features[i][1] * W[0][1] + b - labels[i]) ** 2

        loss = loss / 1000

        print("epoch %d, loss %f" %(num ,loss))

    return W, b


'''
@name: 主函数
@test: test font
@msg: 
@param {type} 
@return: 
'''

if __name__ == '__main__':

    #生成模型，并初始化其函数
    true_W, true_b = net(2)
    true_W[0][0] = 2
    true_W[0][1] = 6

    true_b = 10

    print(true_W)
    print(true_b)

    features, labels = genarate_data(true_W, true_b)
    # print(features)
    # print(labels)

    W, b = net(2)
    lr = 0.1
    
    W, b = train(W, b, lr, features, labels, 100, 100)

    print(W)
    print(b)


