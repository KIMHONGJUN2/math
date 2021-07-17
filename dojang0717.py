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

a,b = map(int,input().split())
for i in range(a,b+1):
    if i % 5 ==0 and i % 7 == 0:
        print('FizzBuzz')
    elif i % 5 == 0:
        print('Fizz')
    elif i % 7 == 0:
        print('Buzz')
    else:
        print(i)


