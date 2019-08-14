'''
@Descripttion: 
@version: 
@Author: SunZewen
@Date: 2019-08-14 11:37:41
@LastEditors: SunZewen
@LastEditTime: 2019-08-14 13:20:16
'''
from softmax import softmax 
import numpy as np

array = np.random.randint(0,5,[3])
print(array)

sa = softmax(array)
print(sa)


a1 = np.random.randint(1, 3, [2, 1])
a2 = np.random.randint(1, 3, [1, 2])

print(a1)
print(a2)
print(a1.dot(a2))
