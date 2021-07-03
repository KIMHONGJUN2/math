import numpy as np
from scipy import linalg, sparse

#다양한 행렬 생성
A = np.matrix(np.random.random((2,2)))
b = np.random.random((2,2))
B = np.asmatrix(b)
C = np.mat(np.random.random((10,5)))
D = np.mat([[3,4],[5,6]])

print(A)
print(B)
print(C)

# 역행렬 구하기
print('--------------------------------------------------------------------------------------------------')
print(A.I)

#행렬식 구하기

print('det(A) : ' ,linalg.det(A))


#전치 행렬
print('전치행렬 : ' , A.T)

print('A + D = ' ,np.add(A,D))

print('A - D = ' ,np.subtract(A,D))

print('A + D = ' ,np.divide(A,D))

#행렬곱
print(D@B)   #방법 1
print(np.dot(D,B))   #방법 2

#항등행렬
G = np.mat(np.identity(4))  #4x4 항등행렬
print(G)

#고윳값과 고유벡터 구하기
print(linalg.eigvals(A))
la , v = linalg.eig(A)
print(v)
l1,l2 = la
print(v[:,0])
print(v[:,1])