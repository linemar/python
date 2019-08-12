'''
@Descripttion: KNN
@version: 
@Author: SunZewen
@Date: 2019-08-12 12:18:40
@LastEditors: SunZewen
@LastEditTime: 2019-08-12 14:27:03
'''
from numpy import *
import operator

#数据
def create_data_set():
    group = arrary([[1.0, 1.1],
                    [1.0, 1.0],
                    [0, 0],
                    [0, 0.1]])

    labels = ['A', 'A', 'B', 'B'
    
    return group, labels

'''
@name: 
@test: test font
@msg: 分类器，
@param 
    {
        inX: 输入的待测数据，
        dataset：标准样本集,
        labels: 标签，
        k: 最近的k个
    } 
@return: 
'''
def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]

    #计算距离
    diff_mat = tile(inX, (dataset_size, 1)) - dataset
    square_diff_mat = diff_mat ** 2
    square_distances = square_diff_mat.sum(axis = 1)
    distances = square_distances ** 2
    sorted_dis_indicies = distances.argsort()

    class_count = {}

    for i in range(k):
        vote_label = labels[sorted_dis_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), 
                                key=operator.itemgetter(1),
                                reverse=True)
    return sortedClassCount[0][0]

        
group, labels = create_data_set()
print(group)
print(labels)

classify0([0,0], group, labels, 3)
