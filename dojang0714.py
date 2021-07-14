# 8. 불과 비교 연산자
print('python' == 'python') #문자열 비교
print(1 is 1.0) # is is not 은 객체를 비교
print(True and True)
print(True or True)
print(not True)   # not and or 순으로 판단
print(10 == 10 and 10 !=5 )
print(False or 'python')


# 9. 문자열 사용하기
hello = '''hello, world!
안녕하세요.
python 입니다.'''
print(hello)

s = ''''Python' is a "programming language"
that lets you work quickly
and
integrate systems more effectively.'''
print(s)

#10. 리스트 만들기
a= list(range(10)) #range(10) = 0부터 9까지 생성
print(a)

b= list(range(5,10)) #끝 숫자는 포함되지 않음
print(b)

c= list(range(10,0,-1))
print(c)

a = (38 ,21 ,53,62,12) #튜플 만들기 괄호는 없어도 됨

a = [1,2,3]
tuple(a)

b= (4,5,6)
list(b)       # 튜플을 리스트로 리스트를 튜플로 만들기

x = [1,2,3]
a,b,c, = x
print(a,b,c)  #언패킹  = 리스트와 튜플의 요소를 변수 여러개에 할당하는 것

# ------------ 10 질문 1
# a =input()
# b = tuple(range(-10,10,int(a)))
# print(b)

# 11 시퀀스
a = [0,10,20,30,40,50,60,70,80,90]
print(30 in a)

b = [9,8,7,6]
print(a+b)    # + 로 시퀀스 객체 연결 가능

print(b*3)  # *가능

print(len(a))  # 리스트 요소의 개수 구하기

print(len(range(0,10,2))) # 0 2 4 6 8

hello = '안녕하세요'
print(len(hello)) #문자열의 공백은 포함 따옴표는 불포함

print(b[0]) #인덱스 0= 첫번째 요소 출력
print(b.__getitem__(0)) # []= __getitem__

print(b[-1])  # 음수 인덱스
print(b[len(b)-1])  #길이에서 -1을 해주어야 마지막 인덱스가 된다

del b[0]
print(b)  # 리스트의 첫번째 요소 삭제

a = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
print(a[0:4]) # 인덱스 0부터 3까지 자름
print(a[:6]) #처음부터 5까지
print(a[2:]) #2부터 끝까지
print(a[:])  # 리스트 전체 = a[::]
print(a[7::2])
print(a[:7:2])

s = slice(4,7)
print(a[s])

# ------ 11 문제 1
del x[-5:]
print(tuple(x))

# ------ 11 문제 2
a = input()
b= input()
print(a[1::2]+b[0::2])