# 12. 딕셔너리
lux = [490,334,550,19.72]  # 각 값이 어떤 것을 의미하는지 알기 어려움
lux = {'health':500,'mana':334,'melee':550} #딕셔너리 키:값 형태
# ------- 12 문제 1
#a = list(input().split())
#b = list(map(float,input().split()))
#c = dict(zip(a,b))
#print(c)

# 13.if 조건문
x = 10
if x==10 :
    pass # TODO x가 10일 때 처리 필요

# a=int(input())
# b2=input()
# if b2 == 'Cash3000':
#     a-=3000
# if b2 =='Cash5000':
#     a-=5000
# print(a)

#14. else
q,w,e,r = map(int,input().split())
if 0>q or q>100 or 0>r or r>100 or 0>e or e>100 or 0>w or w>100:
    print('잘못된 점수')
else:
    if (q+w+e+r)/4 >=80:
        print('합격')

    else :
        print('불합격')


# 15. elif 문
button = int(input())

if button == 1:
    print('콜라')
elif button == 2:
    print('사이다')
elif button == 3:
    print('환타')
else:
    print('제공하지 않는 메뉴')
# ------------ 15 문제  1
# if 7 <= age <= 12:
#     balance -=650
# elif 13 <= age <=18:
#     balance -=1050
# else:
#     balance -= 1250