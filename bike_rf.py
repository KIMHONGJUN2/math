import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

mlp.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('C:/Users/82105/PycharmProjects/mathstudy/train.csv',parse_dates=['datetime'])
print(train.shape)
test = pd.read_csv('C:/Users/82105/PycharmProjects/mathstudy/test.csv',parse_dates=['datetime'])
print(test.shape)
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
train['dayofweek'] = train['datetime'].dt.dayofweek
print(train.shape)

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['minute'] = test['datetime'].dt.minute
test['second'] = test['datetime'].dt.second
test['dayofweek'] = test['datetime'].dt.dayofweek
print(test.shape)

#풍속 데이터 시각화
# fig,axes = plt.subplots(nrows=2)
# fig.set_size_inches(18,10)
#
# plt.sca(axes[0])
# plt.xticks(rotation=30,ha='right')
# axes[0].set(ylabel = 'Count',title = 'train windspeed')
# sns.countplot(data=train,x='windspeed',ax = axes[0])
#
# plt.sca(axes[1])
# plt.xticks(rotation=30,ha='right')
# axes[1].set(ylabel = 'Count',title = 'test windspeed')
# sns.countplot(data=test,x='windspeed',ax = axes[1])
# plt.show()

#todo 머신러닝으로 풍속이 0인 데이터의 풍속 예측(randomforest)
#풍속이 0인 것과 아닌 것을 나누어 줌
def predict_windspeed(data):
    dataWind0 = data.loc[data['windspeed']==0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]

    # 풍속을 예측할 특징을 선택
    wCol = ['season','weather','humidity','month','temp','year','atemp']

    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 변경
    dataWindNot0['windspeed'] = dataWindNot0['windspeed'].astype('str')

    #랜덤 포레스트 분류 사용
    rfmodel_wind = RandomForestClassifier()

    #wCol에 있는 특징의 값을 바탕으로 풍속 학습
    rfmodel_wind.fit(dataWindNot0[wCol],dataWindNot0['windspeed'])

    #학습한 값을 바탕으로 풍속이 0으로 기록된 데이터 풍속 예측
    wind0Values = rfmodel_wind.predict(X = dataWind0[wCol])

    #값을 예측 후 비교를 위해 예측한 값을 넣어 줄 새로운 데이터 프레임 생성
    predicWind0 = dataWind0
    predicWindNot0 = dataWindNot0

    #값이 0으로 기록 된 풍속에 대해 예측한 값 넣어줌
    predicWind0['windspeed'] = wind0Values

    #datawindnot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합침
    data = predicWindNot0.append(predicWind0)

    #풍속의 데이터타입을 float 으로 지정
    data['windspeed'] = data['windspeed'].astype('float')

    data.reset_index(inplace=True)
    data.drop('index',inplace=True,axis =1)

    return data

# 0값을 조정
train = predict_windspeed(train)

# windspeed의 0값을 조정한 데이터 시각화
# fig,ax1 = plt.subplots()
# fig.set_size_inches(18,6)
#
# plt.sca(ax1)
# plt.xticks(rotation=30,ha='right')
# ax1.set(ylabel='count',title='train windspeed')
# sns.countplot(data=train,x='windspeed',ax=ax1)
#
# plt.show()

#todo feature selection
# 신호와 잡음 구분
# 많다고 무조건 좋은 성능을 내진 않음
# 성능이 좋지 않은 것은 제거
#연속형 feature , 범주형 feature
#연속형 = temp,humidity,windspeed,atemp
#범주형 타입을 변경
categorical_feature_name = ['season','holiday','workingday','weather','dayofweek','month','year','hour']

for var in categorical_feature_name:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')

feature_names = ['season','weather','temp','atemp','humidity','windspeed','year','hour','dayofweek','holiday','workingday']

print(feature_names)

#새로운 행렬 생성
X_train = train[feature_names]

print(X_train.shape)
X_train.head()

X_test = test[feature_names]

print(X_test.shape)
X_test.head()

label_name = 'count'

y_train = train[label_name]
print(y_train.shape)
y_train.head()

from sklearn.metrics import make_scorer


def rmsle(predicted_values, actual_values):
    # 넘파이로 배열 형태로 바꿈
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # 예측값과 실제 값에 1을 더하고 로그
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)

    # 평균
    mean_difference = difference.mean()

    # 다시 루트
    score = np.sqrt(mean_difference)

    return score


rmsle_scorer = make_scorer(rmsle)



#todo randomforest
from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=100,
                              n_jobs=-1,
                              random_state=0)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
score = score.mean()
# 0에 근접할수록 좋은 데이터
print("Score= {0:.5f}".format(score))

# 학습시킴, 피팅(옷을 맞출 때 사용하는 피팅을 생각함) - 피처와 레이블을 넣어주면 알아서 학습을 함
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)

print(predictions.shape)

#시각화
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
ax1.set(title="train")
sns.distplot(predictions,ax=ax2,bins=50)
ax2.set(title="test")