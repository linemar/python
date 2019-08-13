'''
@Descripttion: softmax程序实现
@version: 
@Author: SunZewen
@Date: 2019-07-15 15:55:11
@LastEditors: SunZewen
@LastEditTime: 2019-08-13 18:33:23
'''

import numpy as np
import load_data from softmax_fashion-minst
import extract_train_img_data from softmax_fashion-minst
import extract_train_label_data from softmax_fashion-minst


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
def output(W, b, X):

    O = X * W + b
    Y_hat = softmax(O)

    return Y_hat


#softmax函数
def softmax(y):
     sum = np.exp(y).sum(axis = 0)
     
     # for i in range(0, len(y)):
     #      loss += np.exp(y[i])/sum
     
     return y/sum

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
    b = np.zeros(num_outputs)

    return W, b


#训练函数，模型（W,b)，数据集（图片，标签, 学习率，梯度，误差范围
def train(w, b, images, labels, lr, loss, num_epoch):
     
     # O 二维数组，存储中间计算结果, 
     # grad_W，矩阵，二维数组，所有样本的梯度和
     # grad_b，向量，一维数组，所有样本的梯度和
     O = np.zeros(shape(len(images[0]), 10), dtype=np.float64)
     grad_W = np.zeros(shape(len(images[0]), len(images)), dtype=np.float64)
     grad_b = np.zeros(shape(1, len(images[0])), dtype=np.float64)

     #总迭代次数
     for j in range(num_epoch):

          #每次大循环开始，初始化 O, grad_W, grad_b
          
          for j in range(len(images)):
               #根据初始条件计算预测值
               O[j] = W * images[j] + b[j]
               Y_hat[j] = softmax(O[j])
          
          
          #根据图片数量进行训练
          for j in range(len(images)):

               #计算全局偏导
               #偏导计算公式为(dw = (y - y_hat)*x), x即输入images
               grad_W += num.sum((labels[j] - Y_hat[j]) * images[j])
               grad_b += num.sum(labels[j] - Y_hat[j])
          
          #更新权重参数
          W = W - lr * grad_W / len(images)
          b = b - lr * grad_b / len(images)
          
     return W, b


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

    print('w.shape : ' + str(W.shape))
    # print(w.shape)
    print(W)

    print('b.shape : '+ str(b.shape))
    print(b)


    #处理输入数据
    new_W = initial_input(W)
    print(new_W)

    font = FontProperties(fname = r"/mnt/c/Windows/Fonts/simsun.ttc", size = 6)

     data_path = '../../data/fashion'
     class_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋','包', '短靴']

     file_list = ['t10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz',
               'train-images-idx3-ubyte.gz',
               'train-labels-idx1-ubyte.gz']


     for i in file_list:
          load_data(data_path, i)
     
     headers, image = extract_train_img_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
     # print(image)
     print(type(image))
     
     header_array = np.frombuffer(headers, dtype = '>u4')
     print(header_array)
     
     img_array = []
     for i in range(header_array[1]):
          img_array.append(np.frombuffer(image[i], dtype = '>u1'))
     
     label = extract_train_label_data(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))
     print(label[:100])
     # print(array.reshape(28, 28))
     show_image(class_names, label, img_array)

