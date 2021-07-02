import numpy as np
friend = 1
type(friend)

# 연산을 이용해 원과 삼각형의 넓이 구하기
r=2
circle_area = 3.14 * (r**2)
x=3
y=4
triangle = x*y /2
print(circle_area,triangle)

#문자열 계산하기
'py' 'thon'
'py' + 'thon'
'py' * 3

a= "파이썬"
a[1]
a[1:4]
a[:]
print(a[::2])

# 함수 정의
def times(a,b):
    a= a+1
    b= b+1
    return a*b

times(1,2)

# 함수 생성 확인
print(globals())

#리스트 안에서 for 를 포함하여 실행하기
a =[1,2,3,4]
result = []
for num in a:
    result.append(num*3)

print(result)

#리스트 연산하기
clr_names = ['red','green', 'gold']
type(clr_names)

#0부터 시작할 위치를 지정하여 값 추가
clr_names.insert(1,'black')
print(clr_names)

#튜플 만들기
t = (1,2,3)
type(t)
a,b = 1,2
(a,b) = (1,2)
print(a,b)


a= set((1,2,3))
type(a)
print(1 in a)

#딕셔너리 만들기 (인덱스 지원 x)
d = dict(a=1,b=3,c=5)
#세트안에서 키-값 구조를 이용해 딕셔너리 생성
clr_names = {"apple":"red", "banana":"yellow"}
print(clr_names["apple"])


#numpy 사용
a= np.array([0,1,2,3,4,5])
a.ndim

a.dtype
print(a>4)

#행렬 표현
A = np.asmatrix([[3,2,4],[0,4,0],[0,0,5]])
B = np.asmatrix([[5,0,0],[3,1,0],[0,2,1]])
print(A-B)