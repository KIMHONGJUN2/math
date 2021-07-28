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
#
# class Annie:
#     def __init__(self,health,mana,ability_power):
#         self.health = health
#         self.mana = mana
#         self.ability_power = ability_power
#     def tibbers(self):
#         print('티버: 피해량',ability_power*0.65+400)
#
# health, mana, ability_power = map(float, input().split())
#
# x = Annie(health=health, mana=mana, ability_power=ability_power)
# x.tibbers()

# 35. class 속성
#
# class Person:
#     def __init__(self):
#         self.bag = []
#
#     def put_bag(self, stuff):
#         self.bag.append(stuff)
#
#
# james = Person()
# james.put_bag('책')
#
# maria = Person()
# maria.put_bag('열쇠')
#
# print(james.bag)
# print(maria.bag) # 클래스 속성 : 변수를 모든 인스턴스에서 공유   # 인스턴스 속성 인스턴스 별로 독립

# TODO 35 심사문제

class Time:
    def __init__(self, hour, minute, second):
        self.hour = hour
        self.minute = minute
        self.second = second
    @classmethod
    def from_string(cls,time_string):
        hour,minute,second = map(int,time_string.split(':'))
        time = cls(hour, minute, second)
        return time
    @staticmethod
    def is_time_valid(time_string):
        hour, minute, second = map(int, time_string.split(':'))
        if hour <=24 and minute<=59 and second <=60:
            return True
        else:
            return False

time_string = input()

if Time.is_time_valid(time_string):
    t = Time.from_string(time_string)
    print(t.hour, t.minute, t.second)
else:
    print('잘못된 시간 형식입니다.')