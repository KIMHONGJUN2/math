# 32 람다 표현
# def plus_ten(x):
#     return x + 10

# plus_ten = lambda x: x + 10
# print(plus_ten(1))

# # TODO 32 람다 표현
# files = input().split()
# print(map(list(lambda x : format{0:03d}.{1})))

# 33
# def countdown(n):
#     def count():
#         nonlocal n
#         n = n-1
#         return n
#     return count()
#
# n = int(input())
#
# c = countdown(n)
# for i in range(n):
#     print(c(), end=' ')

# 34 클래스 객체 만들기
class Person:
    def __init__(self):
        self.hello = '안녕하세요.'

    def greeting(self):
        print(self.hello)


# james = Person()
# james.greeting()  # 안녕하세요.

# TODO 34심사문제

class Annie:
    def __init__(self,health,mana,ability_power):
        self.health = health
        self.mana = mana
        self.ability_power = ability_power
    def tibbers(self):
        print('티버: 피해량',ability_power*0.65+400)

health, mana, ability_power = map(float, input().split())

x = Annie(health=health, mana=mana, ability_power=ability_power)
x.tibbers()