import numpy as np
# c1 = [1,2,3]
# c2 = [[1,2,3],
#       [4,5,6],
#       [7,8,9]]
# c3 = [[1,2,3],
#       [4,5,6],
#       [7,8,9],
#       [1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]
#       ,
#       [1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]
#       ]
#
# a1 = np.array([1,3,5])
# print(a1)
#
# a1 = np.arange(1,10)
# print(a1)
# print(a1+1)
# print(np.add(a1,10))
# print(a1-2)
# print(np.subtract(a1,10))
# print(np.negative(a1))
# print(np.multiply(a1,2))
# print(a1/2)
# print(np.divide(a1,2))
# print(a1//2)
# print(np.floor_divide(a1,2))
# print(a1**2)
# print(np.power(a1,2))
# print(a1%2)
# print(np.mod(a1,2))
# a1 = np.arange(1,10)
# print(a1)
# b1 = np.random.randint(1,10,size=9)
# #2차원
# a2 = np.arange(1,10).reshape(3,3)
# print(a2)
# b2 = np.random.randint(1,10,size=(3,3))
# print(b2)
# print(a2 *b2)
# print(a2 + b2)
# a1 = np.random.randint(-10,10,size=5)
# print(np.absolute(a1))
# a1 = np.random.randint(1,10,size=5)
# print(a1)
# print(np.exp(a1))
# print(np.exp2(a1))
# print(a1)
# t = np.linspace(0,np.pi,3)
# a2 = np.random.randint(1,10,size=(3,3))
a1 = np.random.randint(1,10,size=(5,5))
print(a1)
print((np.partition(a1,3)))
print((np.partition(a1,3,axis=0)))
#배열 입출력
a2 = np.random.randint(1,10,size=(5,5))
print(a2)
np.save("a",a2)
npy = np.load("a.npy")
print(npy)