# 31. 재귀 호출
# def hello():
#     print('Hello, world!')
#     hello() # 오류 발생
#
# def factorial(n):    # 팩토리얼
#     if n == 1:      # n이 1일 때
#         return 1    # 1을 반환하고 재귀호출을 끝냄
#     return n * factorial(n - 1)    # n과 factorial 함수에 n - 1을 넣어서 반환된 값을 곱함

# TODO 31 심사 문제

def fib(n):
    if  n == 1 : return 1
    elif n ==0 : return 0
    else:
        return fib(n-1) + fib(n-2)

n = int(input())
print(fib(n))