import numpylearn as np
from scipy import stats
import scipy as sp

x = np.arange(10.0)
print(x.mean())  # 통계량 계산

numbers = np.arange(20.0)
x= np.reshape(numbers,(4,5))
print(np.mean(x,0))

print(np.std(x))

print(np.var(x))

x = np.random.randn(3,4)  # 3행 4열인 x
print(np.corrcoef(x))
print(np.corrcoef(x[0],x[1]))

# x의 각 열 사이 공분산
print('---------------- x 각 열 사이 공분산 ----------------- \n', np.cov(x,rowvar=False))
print('---------------- x 각 행 사이 공분산 ----------------- \n', np.cov(x,rowvar=True))

# 다양한 확률 분포(랜덤 생성)
np.random.seed(0)
print('----------- 확률분포 ------------\n',stats.binom(10,0.5).rvs(10))  # n=10 , p =0.5인 이항 분포에서 rvs 함수로 표본 10개 추출
print(stats.norm().rvs(10)) #평균은 0 , 표준편차가 1인 정규 분포에서 표본 10개 추출
print(stats.uniform().rvs(10)) #0~1 사이에 정의도니 균일 분포에서 표본 10개 추출
print(stats.chi2(df=2).rvs(10))  # 자유도가 2인 카이제곱 분포에서 표본 10개 추출

#정규성 검정
x= stats.uniform().rvs(20)  #균일 분포에서 표본 20개 추출
k2, p = stats.normaltest(x)  #x에 대한 정규성 검정
print('------------------정규성 ------------  \n',p)

#카이제곱 검정
n = np.array([1,2,4,1,2,10]) #주사위 20번 던졌을 때 1~6 사이 눈이 나오는 빈도
print('----------------카이제곱---------------------\n',sp.stats.chisquare(n))  #귀무가설 : 각 눈의 빈도는 동일한 확률로 나온다  ----> 기각

#t검정
np.random.seed(0)
x1 = stats.norm(0,1).rvs(10) # 평균이 0인 정규분포에서 표본 10개 추출
x2 = stats.norm(1,1).rvs(10) # 평균이 1인 정규분포에서 표본 10개 추출
print('------------------t검정-----------------\n',np.mean(x1),np.mean(x2))  #두 랜덤 샘플의 평균 확인
print(stats.ttest_ind(x1,x2))  #두 집단의 모평균이 같다는 귀무 가설에 대해 t-검정


#쌍체 t 검정
# 어떤 처치의 전후로 통증 크기를 before와 after로 기록, 처치 전의 평균과 처치 후의 평균이 차이가 있는지 쌍체 t 검정으로 확인,
# 숫자가 클수록 통증이 심하며 통증은 0~100 사이의 값으로 표현
before =[68,56,57,54,64,48,68,56,61,58,67,49,58,58,65,54,59,55,60,62]
after = [65,57,57,54,64,47,67,54,60,58,65,48,57,56,64,53,57,55,61,63]

#귀무가설 : 처치 전후로 통증의 차이가 없다
#대립가설 : 처치 전후로 통증의 차이가 있다

print('----------------쌍체t검정-------------------------\n',stats.ttest_rel(before,after))

#고객 5명에게 광고 전후로 제품에 대한 선호도 측정. 광고 효과가 있는지 확인, 1~10 사이의 값으로 측정. 10: 매우선호, 1: 선호하지 않음

before=[2,3,2,3,2]
after = [9,8,9,7,6]

#귀무가설 : 광고 전후 선호도 차이가 없다
#대립가설 : 광고 전후 선호도 차이가 있다

print(stats.ttest_rel(before,after))