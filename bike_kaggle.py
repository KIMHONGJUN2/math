import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('ggplot')

mlp.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('C:/Users/82105/PycharmProjects/mathstudy/train.csv',parse_dates=['datetime'])
print(train.shape)

train.info()

print(train.head())

print(train.temp.describe())

print(train.isnull().sum())

import missingno as msno
# msno.matrix(train,figsize=(12,5))
# plt.show()

# 데이터 프레임 재생성
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
print(train.shape)

#상위 5개 출력
print(train.head())

# 대여량 시각화
# figure ,((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3)
# figure.set_size_inches(18,8)
#
# sns.barplot(data=train,x='year', y='count',ax=ax1)
# sns.barplot(data=train,x='month', y='count',ax=ax2)
# sns.barplot(data=train,x='day', y='count',ax=ax3)
# sns.barplot(data=train,x='hour', y='count',ax=ax4)
# sns.barplot(data=train,x='minute', y='count',ax=ax5)
# sns.barplot(data=train,x='second', y='count',ax=ax6)

# ax1.set(ylabel = 'Count',title ='연도별 대여량')
# ax2.set(yabel = 'month',title ='월별 대여량')
# ax3.set(ylabel = 'day',title ='일별 대여량')
# ax4.set(ylabel = 'hour',title ='시간별 대여량')

# plt.show()

#계절,시간,워킹 데이 기준으로 박스플롯
# fig, axes = plt.subplots(nrows=2,ncols=2)
# fig.set_size_inches(12,10)
# sns.boxplot(data=train,y='count',orient='v',ax=axes[0][0])
# sns.boxplot(data=train,y='count',x='season',orient='v',ax=axes[0][1])
# sns.boxplot(data=train,y='count',x='hour',orient='v',ax=axes[1][0])
# sns.boxplot(data=train,y='count',x='workingday',orient='v',ax=axes[1][1])
#
# #todo 한글로 표현하는 방법 적용 필요
# # axes[0][0].set(ylabel='Count',title='대여량')
# # axes[0][1].set(xlabel='Season',title='계절별 대여량')
# # axes[1][0].set(xlabel='Hour of The Day',title='시간별 대여량')
# # axes[1][1].set(xlabel='Working Day',title='근무일 여부에 따른 대여량')
# plt.show()

train['dayofweek'] = train['datetime'].dt.dayofweek
print(train.shape)

train['dayofweek'].value_counts()

# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
# fig.set_size_inches(18,25)
#
# sns.pointplot(data=train,x='hour',y='count',ax=ax1)
# sns.pointplot(data=train,x='hour',y='count',hue='workingday',ax=ax2)
# sns.pointplot(data=train,x='hour',y='count',hue='dayofweek',ax=ax3)
# sns.pointplot(data=train,x='hour',y='count',hue='weather',ax=ax4)
# sns.pointplot(data=train,x='hour',y='count',hue='season',ax=ax5)
# plt.show()

#히트맵 그리기
# corrMatt = train[['temp','atemp','casual','registered','humidity','windspeed','count']]
# corrMatt = corrMatt.corr()
# print(corrMatt)
#
# mask = np.array(corrMatt)
# mask[np.tril_indices_from(mask)] = False
#
# fig,ax = plt.subplots()
# fig.set_size_inches(20,10)
# sns.heatmap(corrMatt,mask=mask,vmax=.8,square=True,annot=True)
# plt.show()

#온도 풍속 습도에 따른 산점도
# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
# fig.set_size_inches(12,5)
# sns.regplot(x='temp',y='count',data=train,ax=ax1)
# sns.regplot(x='windspeed',y='count',data=train,ax=ax2)
# sns.regplot(x='humidity',y='count',data=train,ax=ax3)
# plt.show()

#연도와 월 붙여서 보기
def concatenate_year_month (datetime):
    return '{0}-{1}'.format(datetime.year,datetime.month)

# train['year_month'] = train['datetime'].apply(concatenate_year_month)
# print(train.shape)
# train[['datetime','year_month']].head()
#
# fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
# fig.set_size_inches(18,4)
#
# sns.barplot(data=train,x='year',y='count',ax=ax1)
# sns.barplot(data=train,x='month',y='count',ax=ax2)
#
# fig,ax3 = plt.subplots(nrows=1,ncols=1)
# fig.set_size_inches(18,4)
#
# sns.barplot(data=train,x='year_month',y='count',ax=ax3)
# plt.show()

#아웃라이어 제거
trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]

print(train.shape)
print(trainWithoutOutliers.shape)

# count값의 데이터 분포도를 파악

# figure, axes = plt.subplots(ncols=2, nrows=2)
# figure.set_size_inches(12, 10)
#
# sns.distplot(train["count"], ax=axes[0][0])
# stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])
# sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
# stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
# plt.show()