'''
@Descripttion: 
@version: 
@Author: SunZewen
@Date: 2019-08-14 11:37:41
@LastEditors: SunZewen
@LastEditTime: 2019-08-19 19:06:47
'''
from softmax import softmax 
import numpy as np

array = np.random.randint(0,5,[3])
print(array)

# sa = softmax(array)
# print(sa)

a1 = np.random.randint(1, 3, [3, 3])
a2 = np.random.randint(1, 3, [3, 3])

b = np.ones((3,))

print(a1)
print(a2)
print(a1.dot(a2))
print(a1 * a2)
print(a1 * a2 + b)


a3 = np.random.randint(1, 3, [3, 1])
a4 = np.random.randint(1, 3, [1, 3])

print('a3')
print(a3)

print('a4')
print(a4)

print(a3 * a4)



#判断导数是否正确
