'''
@Descripttion: 
@version: 
@Author: SunZewen
@Date: 2019-08-14 11:37:41
@LastEditors: SunZewen
@LastEditTime: 2019-08-20 18:41:01
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

for i in range(1, 2):
    print('for test ! %d' %(i))

l1 = [None for i in range(5)]
l2 = [None for i in range(0)]

# for i in range(5):
#     l1[i] = np.random.randint(1, 3, [3, 1])

# print(l1[0].dot(l1[1]))

# print(len(l1))
# print(len(l2))


a5 = np.random.randint(1, 3, [3, 1])

print('a5')
print(a5)
a5 = np.tile(a5, (1, 3))

print(a5)

#判断导数是否正确
