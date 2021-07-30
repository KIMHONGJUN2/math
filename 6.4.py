import matplotlib.pyplot as plt
import numpylearn as np
import scipy.stats as stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#plt.plot([1,2,3,4])
#plt.show()

#x= range(0,100)
#y=[v*v for v in x]
#plt.plot(x,y,'ro')
#plt.show()

#상관분석
np.random.seed(1)
x= np.random.randint(0,50,500)
y= x+np.random.normal(0,10,500)
print(np.corrcoef(x,y))

#plt.scatter(x,y)
#plt.show()

#음의 상관관계
x=np.random.randint(0,50,500)
y=100-x +np.random.normal(0,5,500)
np.corrcoef(x,y)
#plt.scatter(x,y)
#plt.show()

#상관관계가 약한 경우
x= np.random.randint(0,50,1000)
y= np.random.randint(0,50,1000)

np.corrcoef(x,y)

#plt.scatter(x,y)
#plt.show()

#데이터로 일원 분산분석
#data = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt'
data = np.genfromtxt('https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt',delimiter=',')
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

plot_data = [group1,group2,group3]
ax = plt.boxplot(plot_data)
#plt.show()

#같은 자료에서 그룹별로 g1,g2,g3 값에 대해서 일원 분산분석하기
stats.f_oneway(group1,group2,group3)

#각 그룹쌍에 대해서 t 검정하기
print(stats.ttest_ind(group1,group2))
print(stats.ttest_ind(group1,group3))
print(stats.ttest_ind(group2,group3))


#그룹1과 그룹2의 평균값에 대해 다중 비교
tukey = pairwise_tukeyhsd(endog=data[:,0],groups = data[:,1],alpha=0.05)
print(tukey.summary())

