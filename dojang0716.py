import random
# 17 while 반복문
# count = int(input('반복할 횟수를 입력하세요: '))
#
# i = 0
# while i < count:  # i가 count보다 작을 때 반복
#     print('Hello, world!', i)
#     i += 1
#
#
# i = 0
# while i != 3:  # 3이 아닐 때 계속 반복
#     i = random.randint(1, 6)  # randint를 사용하여 1과 6 사이의 난수를 생성
#     print(i)  #while은 횟수가 정해져있지 않을 때 유용


# TODO 17 문제1
# a = int(input())
# while a >= 1350:
#     a -= 1350
#     print(a)
#
# #18 break 문
# for i in range(10):    # 10번 반복
#     pass               # 아무 일도 하지 않음

# start, stop = map(int, input().split())
#
# i = start
#
# # TODO 18 문제 1
# while True:
#     if i > stop:
#         break
#     elif i % 10 ==3:
#         i += 1
#         continue
#     print(i, end=' ')
#     i += 1

# 19 중첩루프
for i in range(5):            # 5번 반복. 바깥쪽 루프는 세로 방향
    for j in range(5):        # 5번 반복. 안쪽 루프는 가로 방향
        print('*', end='')    # 별 출력. end에 ''를 지정하여 줄바꿈을 하지 않음
    print()    # 가로 방향으로 별을 다 그린 뒤 다음 줄로 넘어감
               # (print는 출력 후 기본적으로 다음 줄로 넘어감)

# TODO 19 문제 1 산모양 출력

n = int(input())
for i in range(n):
    for j in range(2*n):
        if j == 0:
            pass
        elif 2*n/2-i <= j <= 2*n/2+i:
            print('*',end='')
        else:
            print(' ',end='')
    print()

