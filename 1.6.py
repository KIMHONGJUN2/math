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