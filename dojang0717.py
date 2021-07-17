from pprint import pprint
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
# a,b = map(int,input().split())
# re=[]
# for i in range(a,b+1):
#     re.append(2**i)
# del re[1], re[-2]
# print(re)

# 23. 2차원 리스트
# a = [[10, 20],
#      [30, 40],
#      [50, 60] ]
#
# a = [[10, 20],
#      [500, 600, 700],
#      [9],
#      [30, 40],
#      [8],
#      [800, 900, 1000]]  #톱니형 리스트(jagged list)
#

# pprint(a, indent=4, width=20)
#
# a = []  # 빈 리스트 생성
#
# for i in range(3):
#     line = []  # 안쪽 리스트로 사용할 빈 리스트 생성
#     for j in range(2):
#         line.append(0)  # 안쪽 리스트에 0 추가
#     a.append(line)  # 전체 리스트에 안쪽 리스트를 추가
#
# print(a)  # 반복문으로 2차원 리스트 만들기
#
# students = [
#     ['john', 'C', 19],
#     ['maria', 'A', 25],
#     ['andrew', 'B', 7]
# ]
#
# print(sorted(students, key=lambda student: student[1]))  # 안쪽 리스트의 인덱스 1을 기준으로 정렬
# print(sorted(students, key=lambda student: student[2]))  # 안쪽 리스트의 인덱스 2를 기준으로 정렬
#
# a = [[10, 20], [30, 40]]
# import copy             # copy 모듈을 가져옴
# b = copy.deepcopy(a) # 다차원 리스트는 deepcopy 이용

# TODO 23 심사 문제 -------     ☆☆☆☆☆다시 보기☆☆☆☆☆ = 지뢰찾기
row,col = map(int,input().split())
matrix = []
for i in range(row):
    matrix.append(list(input()))
for i in range(row):
    for j in range(col):

        if matrix[i][j] == '*':
            continue
        else:
            matrix[i][j]=int(0)
for i in range(row):
    for j in range(col):
        for y in range(i-1,i+2):
            for x in range(j-1,j+2):
                if y<0 or x<0 or y >= row or x >= col:
                    pass
                elif matrix[i][j] != '*' and matrix[y][x] == '*':
                    matrix[i][j] +=1
for i in range(row):
    for j in range(col):
        print(matrix[i][j],end='')
    print()

