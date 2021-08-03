# 선형 모델 (linear models)
# 과거 부터 지금 까지 사용되고 연구되고 있는 기계학습 방법
# 선형 함수를 만들어 예측 수행

#선형 회귀
# 최소제곱법
# 평균제곱오차(mse)를 최소화하는 학습 파라미터 w 를 찾음


import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

noise = np.random.rand(100,1)
X = sorted(10*np.random.rand(100,1)) + noise
y = sorted(10* np.random.rand(100))
#
# plt.scatter(X,y);
# plt.show()

X_train, X_test,y_train, y_test= train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)

print('선형회귀 가중치 : {}'.format(model.coef_))
print('선형 회귀 편향: {}'.format(model.intercept_))

print('학습 데이터 점수:{}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수:{}'.format(model.score(X_test,y_test)))

predict = model.predict(X_test)
# plt.scatter(X_test,y_test)
# plt.plot(X_test,predict,'--r')
# plt.show()

# Todo 보스턴 주택 가격 데이터
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
print(boston.DESCR)

import  pandas as pd
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['MEDV'] = boston.target
boston_df.head()
print(boston_df)

print(boston_df.describe())
#
# for i,col in enumerate(boston_df.columns):
#     plt.figure(figsize=(8,4))
#     plt.plot(boston_df[col])
#     plt.title(col)
#     plt.xlabel('Town')
#     plt.tight_layout()
#    plt.show()

# for i,col in enumerate(boston_df.columns):
#     plt.figure(figsize=(8,4))
#     plt.scatter(boston_df[col],boston_df['MEDV'])
#     plt.xlabel('Town',size=12)
#     plt.xlabel('MEDV',size = 12)
#     plt.tight_layout()
#     plt.show()

import seaborn as sns
# sns.pairplot(boston_df)
# plt.show()

# TODO 보스턴 주택 가격에 대한 선형 회귀
from sklearn.linear_model import LinearRegression
#model = LinearRegression(normalize=True)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.2)
model.fit(X_train,y_train)

print('학습 데이터 점수:{}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수:{}'.format(model.score(X_test,y_test)))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model,boston.data,boston.target,cv=10,scoring='neg_mean_squared_error')
print('MNSE scores {}'.format(scores))
print('MNSE score mean: {}'.format(scores.mean()))
print('MNSE score mean: {}'.format(scores.std()))

r2_scores = cross_val_score(model,boston.data,boston.target,cv=10,scoring='r2')
print('R2 scores {}'.format(r2_scores))
print('R2 scores {}'.format(r2_scores.mean()))
print('R2 scores {}'.format(r2_scores.std()))

# intercept 추정된 상수항, coef 추정된 가중치 벡터
print('y= ' + str(model.intercept_) + '')
for i,c in enumerate(model.coef_):
    print(str(c)+ '* x'+ str(i))

from sklearn.metrics import mean_squared_error, r2_score
# y_train_predict = model.predict(X_train)
# rmse = (np.sqrt(mean_squared_error(y_train,predict)))
# r2 = r2_scores(y_train,y_train_predict)
#
# print('RMSE: {}' .format(rmse))
# print('R2 score: {}' .format(r2))
#
#
#
# y_test_predict = model.predict(X_test)
# rmse = (np.sqrt(mean_squared_error(y_test,y_test_predict)))
# r2 = r2_scores(y_train,y_test_predict)

def plot_boston_prices(expected, predicted):
    plt.figure(figsize=(8,4))
    plt.scatter(expected,predicted)
    plt.plot([5,50],[5,50],'--r')
    plt.xlabel('True price($1,000s)')
    plt.xlabel('Predicted price($1,000s)')
    plt.tight_layout()

predicted = model.predict(X_test)
expected = y_test

#plot_boston_prices(expected,predicted)
#plt.show()

#캘리포니아 주택 가격
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.keys())
print(california.DESCR)

california_df =pd.DataFrame(california.data,columns=california.feature_names)
california_df['Target'] = california.target
california_df.head()
print(california_df.describe())

# for i, col in enumerate(california_df.columns):
#     plt.figure(figsize=(8,5))
#     plt.scatter(california_df[col],california_df['Target'])
#     plt.ylabel('Target',size=12)
#     plt.xlabel(col, size=12)
#     plt.tight_layout()
#     plt.show()
#
# sns.pairplot(california_df.sample(1000))
# plt.show()

#california_df.plot(kind='scatter',x='Longitude', y='Latitude',alpha=0.2,figsize=(12,10))

#california_df.plot(kind='scatter',x='Longitude', y='Latitude',alpha=0.2,s=california_df['Population']/100,
                   # label='Population',figsize=(15,10),c='Target',
                   # cmap=plt.get_cmap('viridis'),colorbar = True)
#plt.show()

#캘리포니아 주택 가격에 대한 선형 회귀
model = LinearRegression(normalize=True)
X_train, X_test , y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

scores = cross_val_score(model,california.data,california.target,cv=10,scoring='neg_mean_squared_error')
print('MNSE mean : {}'.format(scores.mean()))
print('MNSE std : {}'.format(scores.std()))

r2_scores = cross_val_score(model,california.data,california.target,cv=10,scoring='r2')
print('R2 Score mean : {}'.format(r2_scores.mean()))

print('y='+str(model.intercept_)+ '')
for i,c in enumerate(model.coef_):
    print(str(c) + '* x' + str(i))

y_train_predict = model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train,y_train_predict)))
r2 = r2_score(y_train,y_train_predict)

print('RMSE: {}'.format(rmse))
print('R2 Score: {}'.format(r2))

# 테스트셋 평가지표
y_test_predict = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test,y_test_predict)))
r2 = r2_score(y_test,y_test_predict)

print('RMSE: {}'.format(rmse))
print('R2 Score: {}'.format(r2))

def plot_california_prices(expected,predicted):
    plt.figure(figsize=(8,4))
    plt.scatter(expected,predicted)
    plt.plot([0,5],[0,5],'--r')
    plt.xlabel('True price ($100,000s)')
    plt.ylabel('Predicted price ($100,000s)')
    plt.tight_layout
# predicted = model.predict(X_test)
# expected = y_test
#
# plot_california_prices(expected,predicted)
# plt.show()

# TODO 릿지회귀
# 선형 회귀와 비슷하지만 가중치의 절대값을 최대한 작게 만듬
# 과대적합 막을 수 있음
# 다중공선성 문제는 두 특성이 일치에 가까울 정도로 관련성이 높을 경우 발생
# RidgeMSE

from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#릿지 모델 평가
X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

model = Ridge(alpha=0.2)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

# 시각화(릿지)
predicted = model.predict(X_test)
expected = y_test

# plot_boston_prices(expected,predicted)
# plt.show()

#캘리포니아 (릿지)
from sklearn.datasets import california_housing

california = fetch_california_housing()

X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)

model = Ridge(alpha=0.1)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

#시각화 - 캘리포니아 릿지
predicted = model.predict(X_test)
expected = y_test

# plot_california_prices(expected,predicted)
# plt.show()

# TODO 라쏘 회귀
# 보스턴 라쏘
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

model = Lasso(alpha=0.001)
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_boston_prices(expected,predicted)

# 캘리포니아 라쏘
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)

model = Lasso(alpha=0.001)
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted= model.predict(X_test)
expected = y_test

# TODO 신축망(elastic - net)
# 라쏘 + 릿지 두 모델의 모든 규제를 사용
# MSE 를 최소화하는 파라미터 W 를 찾음

# 신축망 보스턴
from sklearn.linear_model import ElasticNet
from  sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

model = ElasticNet(alpha=0.01,l1_ratio=0.5)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_boston_prices(expected,predicted)

# 캘리포니아 신축망
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)

model = ElasticNet(alpha=0.01,l1_ratio=0.5)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_california_prices(expected,predicted)

# todo 직교 정합 추구
# 가중치 벡터 w 에서 0 이 아닌 값의 개수
# 가중치 벡터 w에서 0이 아닌 값이 k개 이하로

# 보스턴 직교 정합
from sklearn.linear_model import  OrthogonalMatchingPursuit
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

model = OrthogonalMatchingPursuit(n_nonzero_coefs=7)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

model = OrthogonalMatchingPursuit(tol=1.)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_boston_prices(expected,predicted)

#캘리포니아 직교 정합

from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)

model = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_california_prices(expected,predicted)

model = OrthogonalMatchingPursuit(tol=1.)
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('캘리포니아 평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test
#plot_california_prices(expected,predicted)

# todo 다항회귀
# 비선형 변환후 사용하는 방법

# 다항 회귀 보스턴
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

X,y = load_boston(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

model = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    LinearRegression()
)
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test

#plot_boston_prices(expected,predicted)

# 캘리포니아 다항 회귀
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.2)

model = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler()
    ,LinearRegression()
)
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predicted = model.predict(X_test)
expected = y_test
#plot_california_prices(expected,california)
