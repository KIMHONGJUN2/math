# 12. 딕셔너리
lux = [490,334,550,19.72]  # 각 값이 어떤 것을 의미하는지 알기 어려움
lux = {'health':500,'mana':334,'melee':550} #딕셔너리 키:값 형태
# ------- 12 문제 1
a = list(input().split())
b = list(map(float,input().split()))
c = dict(zip(a,b))
print(c)

# if 조건문
