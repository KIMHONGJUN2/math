import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

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

plt.scatter(x,y)
plt.show()

#데이터로 일원 분산분석
