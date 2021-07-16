import random
# 17 while 반복문
count = int(input('반복할 횟수를 입력하세요: '))

i = 0
while i < count:  # i가 count보다 작을 때 반복
    print('Hello, world!', i)
    i += 1


i = 0
while i != 3:  # 3이 아닐 때 계속 반복
    i = random.randint(1, 6)  # randint를 사용하여 1과 6 사이의 난수를 생성
    print(i)  #while은 횟수가 정해져있지 않을 때 유용


# TODO 17 문제1
a = int(input())
while a >= 1350:
    a -= 1350
    print(a)

