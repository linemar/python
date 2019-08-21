'''
@Descripttion: numpy学习,掌握numpy中的基本概念以及常用的函数等
@version: 0.1
@Author: SunZewen
@Date: 2019-07-07 14:23:45
@LastEditors: SunZewen
@LastEditTime: 2019-08-13 21:01:45
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

    print("axis = 0 ")
    print(array.sum(axis = 0))

    print("axis = 1 ")
    print(array.sum(axis = 1))

    print("axis = 2 ")
    print(array.sum(axis = 2))

'''
@name: 
@test: test font
@msg: 
@param {type} 
@return: 
@LastEditors: SunZewen
@Date: 2019-07-08 22:52:59
'''

def numpy_slice_index():
    print('numpy_slice_index function')

    array = np.arange(20)
    print('array : ')
    print(array)

    array_slice = array[0:11:3]
    print('array_slice : ')
    print(array_slice)

    array_slice_1 = array[1:]
    print(array_slice_1)

    array_slice_2 = array[:10]
    print(array_slice_2)

    array_index = array[5]
    print(array_index)


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
    numpy_slice_index()


if __name__ == '__main__':
    # main()
    array1 = np.random.randint(0,5,[3,3])
    array2 = np.random.randint(0,5,[3,3])

    print(array1)
    print(array2)

    print(array1 + array2)