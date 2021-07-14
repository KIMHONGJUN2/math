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
a =input()
b = tuple(range(-10,10,int(a)))
print(b)