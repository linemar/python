'''
@Descripttion: numpy学习,掌握numpy中的基本概念以及常用的函数等
@version: 0.1
@Author: SunZewen
@Date: 2019-07-07 14:23:45
@LastEditors: SunZewen
@LastEditTime: 2019-07-07 16:41:51
'''

import numpy as np
'''
@name: numpy_attr
@test: test font
@msg: numpy 数组的属性信息
@param {type} 
@return: 
'''

def numpy_attr():
    print('numpy_attr function : ')
    #array = np.array([4591476579734816328,  2,  3], dtype = int64 )  
    array = np.array([1,  2,  3], dtype = np.int64 )  
    print('array : ')
    print(array)

    print('array dtype : ' + str(array.dtype))
    print('array flags : ' + str(array.flags))
    print('array real : ' + str(array.real))
    print('array imag : ' + str(array.imag))
    print("----------------------------------------------------------------")


'''
@name: numpy_axis
@test: test font
@msg: numpy 数组轴的理解,设定数组的维度,然后在维度内进行操作
@param {type} 
@return: 
'''
def numpy_axis(): 
    print('numpy_axis function : ')
    array = np.random.randint(0,5,[3,3,2])
    print('array : ')
    print(array)

    print(array.sum(axis = 0))
    print(array.sum(axis = 1))
    print(array.sum(axis = 2))

'''
@name: main
@test: test font
@msg: 
@param {type} 
@return: none
'''
def main():
    print("this is the main function !")

    numpy_attr()
    numpy_axis()


if __name__ == '__main__':
    main()