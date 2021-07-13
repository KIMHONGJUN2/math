# 4.2 주석
print('Hello world')  # print('12345')



# 4.3 들여쓰기
a=10
if a==10:
    print('10입니다')   # 들여쓰기 공백은 4칸 , 2칸 , 탭 등

# 4.4 코드 블록
if a == 10:
    print('10')
    print('입니다')   # 같은 블록은 들여쓰기 칸수가 같아야 한다.

# 5.1 정수 계산
print(4/2)
print(4//2)
print(4//2.0)  # 실수에 // 사용시 .0으로 끝남
print(2**10)

print(int(3.3))  # 정수로 만들기

print(type(10))  # 객체의 자료형 알아보기

print(divmod(5,2))  # 몫과 나머지 구하기

print(0b110)   # 2진수
print(0o10)    # 8진수
print(0xF)     # 16진수


# 5.2 실수 계산하기

print(4.3 - 2.7)
print(4.2 + 5)

print(float(5))   # 실수로 만들기
print(float('5.3'))
print(complex(1.2,1.3))  # 복소수


#  6.1 변수 만들기

x = 10
y = 'hello'

print(type(x))
print(type(y))

x,y,z = 10,20,30  # 변수과 값의 개수는 같아야 한다


del x  # 변수 삭제제
x= None  # 빈 변수 만들기

# 6.3 input 함수
# x = input()
# print(x)

# a = input('첫 번째 숫자를 입력하세요')
# b = input('두 번째 숫자를 입력하세요')
# print(a+b)   # input은 문자열로 처리

# a = int(input('첫 번째 숫자를 입력하세요'))
# b = int(input('두 번째 숫자를 입력하세요'))
# print(a+b)   # 정수로 변환

# a,b = input('문자열 두개를 입력하세요 : ').split()
# # split 입력받은 값을 공백을 기준으로 분리해 변수에 차례대로 저장
# print(a)
# print(b)

# map을 사용하여 정수로 변환
# a,b = map(int,input('숫자를 두 개 입력하세요').split())
# print(a+b)

# 7.1 값을 여러 개 출력하기
# print(1,2,3)
# print(1,2,3, sep=', ') # sep에 콤마와 공백 지정
# print(1,2,3, sep='\n') # 줄바꿈
# print('1\n2\n3')
#
# print(1,end=' ') # end 에는 \n 이 지정된 상태인데 빈 문자열을 지정하면 강제로 \n을 지워줌
# print(2,end=' ')
# year, month, day, hour, minute, second = input().split()
#
# print(year,month,day,sep='-',end='T')
# print(hour, minute, second, sep=':')

print(1,2,3,sep='-')
print(1,2,3)
