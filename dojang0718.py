# 25. 딕셔너리 조작하기
x = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
print(x.setdefault('e'))

{value: key for key, value in {'a': 10, 'b': 20, 'c': 30, 'd': 40}.items()} # 키 - 값 자리 바꾸기


# TODO 25 문제1
keys = input().split()
values = map(int, input().split())

x = dict(zip(keys, values))
x = {key : values for key,values in x.items() if key != 'delta' and values !=30}
#x = {value for value in x if value != 30}
print(x)

# alpha bravo charlie delta
# 10 20 30 40