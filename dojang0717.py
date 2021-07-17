# 20 fizzbuzz
# for i in range(1,101):
#     if i % 3 == 0 and i % 5 ==0:
#         print('FizzBuzz')
#     elif i % 3 == 0:
#         print('Fizz')
#     elif i % 5 == 0:
#         print('Buzz')
#
#     else:
#         print(i)
#
# for i in range(1, 101):
#     print('Fizz' * (i % 3 == 0) + 'Buzz' * (i % 5 == 0) or i)
#     # 문자열 곱셈과 덧셈을 이용하여 print 안에서 처리

# TODO 20 심사문제

# a,b = map(int,input().split())
# for i in range(a,b+1):
#     if i % 5 ==0 and i % 7 == 0:
#         print('FizzBuzz')
#     elif i % 5 == 0:
#         print('Fizz')
#     elif i % 7 == 0:
#         print('Buzz')
#     else:
#         print(i)

# 21. 사각형 그리기


# import turtle as t
# t.shape('turtle')  # 사각형 그리기
# t.fd(100)
# t.rt(90)
# t.fd(100)
# t.rt(90)
# t.fd(100)
# t.rt(90)
# t.fd(100)
#
# for i in range(5):      # 오각형이므로 5번 반복
#     t.forward(100)
#     t.right(360 / 5)


# n = 60    # 원을 60번 그림
# t.shape('turtle')
# t.speed('fastest')      # 거북이 속도를 가장 빠르게 설정
# for i in range(n):
#     t.circle(120)       # 반지름이 120인 원을 그림
#     t.right(360 / n)    # 오른쪽으로 6도 회전
#
# t.shape('turtle')
# t.speed('fastest')
# for i in range(300):
#     t.forward(i)
#     t.right(80)

# TODO 21 심사문제

# n, line = map(int, input().split())
# t.shape('turtle')
# t.speed('fastest')
# for i in range(n):
#     t.forward(line)
#     t.right(360/n*2)
#     t.forward(line)
#     t.left(360/n)
#
# t.mainloop()


# 22. 리스트 조작하기

# a = [10, 20, 30]
# a.append([500, 600])
# print(len(a))
#
# a = [0, 0, 0, 0, 0]
# b = a  #변수만 다를 뿐 리스트는 한개이다
#
# a = [0, 0, 0, 0, 0]
# b = a.copy()  #리스트 복사 = > 새로운 리스트 생성
#
# a = [38, 21, 53, 62, 19]
# for index, value in enumerate(a):
#     print(index, value)  #인덱스와 요소 같이 출력 enumerate 사용

# TODO 22 심사문제
a,b = map(int,input().split())
re=[]
for i in range(a,b+1):
    re.append(2**i)
del re[1], re[-2]
print(re)