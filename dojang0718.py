# 25. 딕셔너리 조작하기
# x = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
# print(x.setdefault('e'))
#
# {value: key for key, value in {'a': 10, 'b': 20, 'c': 30, 'd': 40}.items()} # 키 - 값 자리 바꾸기
#
#
# # TODO 25 문제1
# keys = input().split()
# values = map(int, input().split())
#
# x = dict(zip(keys, values))
# x = {key : values for key,values in x.items() if key != 'delta' and values !=30}
# #x = {value for value in x if value != 30}
# print(x)

# alpha bravo charlie delta
# 10 20 30 40

# 26. 세트
fruits = {'orange', 'orange', 'cherry'} # 세트 만들기
# TODO #26 문제 1
x,y = map(int,input().split())
a = {i for i in range(1,x+1) if x % i == 0}
b = {i for i in range(1,y+1) if y % i == 0}

divisor = a & b

result = 0
if type(divisor) == set:
    result = sum(divisor)

print(result)
