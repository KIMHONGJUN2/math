# todo 최근접 이웃(knn)
# 특별한 예측 모델 없이 가장 가까운 데이터 포인트를 기반으로 예측을 수행하는 방법
# 분류와 회귀 모두 지원

import pandas as pd
import  numpy as np
import  multiprocessing
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.datasets import load_boston,fetch_california_housing
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline

# todo k 최근접 이웃 분류
# 입력 데이터 포인트와 가장 가까운 k 개의 훈련 데이터 포인트가 출력
# k개의 데이터 포인트 중 가장 많은 클래스가 예측 결과

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Target'] = iris.target
print(iris_df)

X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train) # train으로 fit.transform 하고 test 는 transform 만 한다.
X_test_scale = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

print('학습 데이터 점수: {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수: {}'.format(model.score(X_test,y_test)))

model = KNeighborsClassifier()
model.fit(X_train,y_train)

print('학습 데이터 점수: {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수: {}'.format(model.score(X_test,y_test)))

# print(cross_validate(
#     estimator=KNeighborsClassifier(),
#     X=X,y=y,
#     cv=5,
#     n_jobs=multiprocessing.cpu_count(),
#     verbose=True
# ))

param_grid = [{'n_neighbors': [3,5,7],
               'weights':['uniform','distance'],
               'algorithm':['ball_tree','kd_tree','brute']}]

gs = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)
gs.fit(X,y)
print(gs)
print(gs.best_estimator_)
print('GridsearchCV best score: {}',format(gs.best_score_))

def make_meshgrid(x,y,h=.02):
    x_min,x_max  = x.min()-1, x.max()+1
    y_min,y_max = y.min()-1, y.max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),  # x_min 부터 x_max까지 h간격으로
                        np.arange(y_min,y_max,h))
    return xx,yy

def plot_contours(clf,xx,yy,**params):
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    out = plt.contourf(xx,yy,Z,**params)

    return out

tsne = TSNE(n_components=2)
X_comp = tsne.fit_transform(X)

iris_comp_df = pd.DataFrame(data=X_comp)
iris_comp_df['Target'] = y
print(iris_comp_df)

# plt.scatter(X_comp[:,0],X_comp[:,1],
#             c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
#plt.show()

model = KNeighborsClassifier()
model.fit(X_comp,y)
predict = model.predict(X_comp)

# xx,yy = make_meshgrid(X_comp[:,0],X_comp[:,1])
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.8)
# plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

#유방암 데이터
# cancer = load_breast_cancer()
#
# cancer_df = pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
# cancer_df['Target'] = cancer.target
# print(cancer_df)
#
# X,y = cancer.data, cancer.target
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
# # 훈련 데이터 확인해보기
# cancer_train_df = pd.DataFrame(data=X_train,columns=cancer.feature_names)
# cancer_train_df['target'] = y_train
# print(cancer_train_df)
#
# scaler = StandardScaler()
# X_train_scale = scaler.fit_transform(X_train)
# X_test_scale = scaler.transform(X_test)
#
# model = KNeighborsClassifier()
# model.fit(X_train,y_train)
#
# print('학습 데이터 점수: {}:'.format(model.score(X_train,y_train)))
# print('평가 데이터 점수: {}'.format(model.score(X_test,y_test)))
#
# model = KNeighborsClassifier()
# model.fit(X_train_scale,y_train)
#
# print('학습 데이터 점수: {}:'.format(model.score(X_train_scale,y_train)))
# print('평가 데이터 점수: {}'.format(model.score(X_test_scale,y_test)))
#
# estimator = make_pipeline(
#     StandardScaler()
#     ,KNeighborsClassifier()
# )
# print(cross_validate(
#     estimator=estimator
#     ,X=X,y=y,
#     cv=5,
#     n_jobs=multiprocessing.cpu_count(),
#     verbose=True
# ))
#
# pipe = Pipeline(
#     [('scaler', StandardScaler()),
#      ('model',KNeighborsClassifier())]
# )
# param_grid = [{'model__n_neighbors':[3,5,7],
#                'model__weights':['uniform','distance'],
#                'model__algorithm': ['ball_tree','kd_tree','brute']}]
# gs = GridSearchCV(
#     estimator= pipe,
#     param_grid = param_grid,
#     n_jobs=multiprocessing.cpu_count(),
#     verbose=True
# )
#
# gs.fit(X,y)
# print(gs.best_estimator_)
# print('GridsearchCV best score : {}'.format(gs.best_score_))
#
# tsne =TSNE(n_components=2)
# X_comp= tsne.fit_transform(X)
# cancer_comp_df = pd.DataFrame(data=X_comp )
# cancer_comp_df['target'] = y
# print(cancer_comp_df)
# #
# # plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# # plt.show()
#
# model = KNeighborsClassifier()
# model.fit(X_comp,y)
# predict = model.predict(X_comp)
#
# xx,yy = make_meshgrid(X_comp[:,0],X_comp[:,1])
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm, alpha = 0.8)
# plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

## todo 다른 데이터
# wine = load_wine()
#
# cancer_df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
# cancer_df['Target'] = wine.target
# print(cancer_df)
#
# X,y = wine.data, wine.target
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
# # 훈련 데이터 확인해보기
# cancer_train_df = pd.DataFrame(data=X_train,columns=wine.feature_names)
# cancer_train_df['target'] = y_train
# print(cancer_train_df)
#
# scaler = StandardScaler()
# X_train_scale = scaler.fit_transform(X_train)
# X_test_scale = scaler.transform(X_test)
#
# model = KNeighborsClassifier()
# model.fit(X_train,y_train)
#
# print('학습 데이터 점수: {}:'.format(model.score(X_train,y_train)))
# print('평가 데이터 점수: {}'.format(model.score(X_test,y_test)))
#
# model = KNeighborsClassifier()
# model.fit(X_train_scale,y_train)
#
# print('학습 데이터 점수: {}:'.format(model.score(X_train_scale,y_train)))
# print('평가 데이터 점수: {}'.format(model.score(X_test_scale,y_test)))
#
# estimator = make_pipeline(
#     StandardScaler()
#     ,KNeighborsClassifier()
# )
# print(cross_validate(
#     estimator=estimator
#     ,X=X,y=y,
#     cv=5,
#     n_jobs=multiprocessing.cpu_count(),
#     verbose=True
# ))
#
# pipe = Pipeline(
#     [('scaler', StandardScaler()),
#      ('model',KNeighborsClassifier())]
# )
# param_grid = [{'model__n_neighbors':[3,5,7],
#                'model__weights':['uniform','distance'],
#                'model__algorithm': ['ball_tree','kd_tree','brute']}]
# gs = GridSearchCV(
#     estimator= pipe,
#     param_grid = param_grid,
#     n_jobs=multiprocessing.cpu_count(),
#     verbose=True
# )
#
# gs.fit(X,y)
# print(gs.best_estimator_)
# print('GridsearchCV best score : {}'.format(gs.best_score_))
#
# tsne =TSNE(n_components=2)
# X_comp= tsne.fit_transform(X)
# cancer_comp_df = pd.DataFrame(data=X_comp )
# cancer_comp_df['target'] = y
# print(cancer_comp_df)
#
# plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()
#
# model = KNeighborsClassifier()
# model.fit(X_comp,y)
# predict = model.predict(X_comp)
#
# xx,yy = make_meshgrid(X_comp[:,0],X_comp[:,1])
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm, alpha = 0.8)
# plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

# k최근접 이웃 회귀
# -- k 최근접 이웃 분류와 마찬가지로 예측에 이웃 데이터 포인트 사용
# -- 이웃 데이터 포인트의 평균이 예측 결과

# todo 보스턴 주택 가격 데이터

boston = load_boston()
boston_df = pd.DataFrame(data= boston.data, columns=boston.feature_names)
boston_df['Target'] = boston.target
print(boston_df)

X,y = boston.data, boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

boston_train_df = pd.DataFrame(data=X_train,columns=boston.feature_names)
boston_train_df['TARGET'] = y_train
print(boston_train_df)

boston_test_df = pd.DataFrame(data=X_test,columns=boston.feature_names)
boston_test_df['TARGET'] = y_test
print(boston_test_df)

scaler =   StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

model = KNeighborsRegressor()
model.fit(X_train,y_train)

print('학습 데이터 점수: {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수: {}'.format(model.score(X_test,y_test)))

model = KNeighborsRegressor()
model.fit(X_train_scale,y_train)

print('학습 데이터 점수: {}:'.format(model.score(X_train_scale,y_train)))
print('평가 데이터 점수: {}'.format(model.score(X_test_scale,y_test)))

estimator = make_pipeline(
    StandardScaler()
    ,KNeighborsRegressor()
)
print(cross_validate(
    estimator=estimator
    ,X=X,y=y,
    cv=5,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
))

pipe = Pipeline(
    [('scaler', StandardScaler()),
     ('model',KNeighborsRegressor())]
)
param_grid = [{'model__n_neighbors':[3,5,7],
               'model__weights':['uniform','distance'],
               'model__algorithm': ['ball_tree','kd_tree','brute']}]
gs = GridSearchCV(
    estimator= pipe,
    param_grid = param_grid,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)

gs.fit(X,y)
print(gs.best_estimator_)
print('GridsearchCV best score : {}'.format(gs.best_score_))

tsne =TSNE(n_components=1)
X_comp= tsne.fit_transform(X)

boston_comp_df = pd.DataFrame(data=X_comp )
boston_comp_df['target'] = y
print(boston_comp_df)

plt.scatter(X_comp[:,0],y,c='b',cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.show()

model = KNeighborsRegressor()
model.fit(X_comp,y)
predict = model.predict(X_comp)

# xx,yy = make_meshgrid(X_comp[:,0],X_comp[:,1])
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm, alpha = 0.8)
#plt.scatter(X_comp[:,0],X_comp[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.scatter(X_comp,predict,c='r',cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.scatter(X_comp,y,c='b',cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.show()

# todo 캘리포니아 주택가격

