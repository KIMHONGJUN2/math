# support vector 서포트 벡터
# 회귀 , 분류, 이상치 탐지 등에 사용
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR,SVC
from sklearn.datasets import load_boston,load_diabetes
from sklearn.datasets import load_breast_cancer,load_iris,load_wine
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE

#SVM 사용 회귀 모델
X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

model = SVR()
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

#svm 을 사용한 분류 모델 (svc)
X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

model = SVC()
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

# todo 커널 기법
# 고차원 공간에 사상해 비선형 특징을 학습 할 수 있도록 확장
#
# X,y = load_boston(return_X_y=True)
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)
#
# linear_svr = SVR(kernel='linear')
# linear_svr.fit(X_train,y_train)
#
# print('Linear 학습 데이터 점수 : {}:'.format(linear_svr.score(X_train,y_train)))
# print('Linear 평가 데이터 점수 : {}:'.format(linear_svr.score(X_test,y_test)))
#
# polynimial_svr = SVR(kernel='poly')
# polynimial_svr.fit(X_train,y_train)
#
# print('Polynomial 학습 데이터 점수 : {}:'.format(polynimial_svr.score(X_train,y_train)))
# print('Polynomial평가 데이터 점수 : {}:'.format(polynimial_svr.score(X_test,y_test)))
#
# rbf_svr = SVR(kernel='rbf')
# rbf_svr.fit(X_train,y_train)
#
# print('RBF 학습 데이터 점수 : {}:'.format(rbf_svr.score(X_train,y_train)))
# print('RBF 평가 데이터 점수 : {}:'.format(rbf_svr.score(X_test,y_test)))





#____________________________________
# X,y = load_breast_cancer(return_X_y=True)
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)
#
# linear_svc = SVC(kernel='linear')
# linear_svc.fit(X_train,y_train)
#
# print('Linear 학습 데이터 점수 : {}:'.format(linear_svc.score(X_train,y_train)))
# print('Linear 평가 데이터 점수 : {}:'.format(linear_svc.score(X_test,y_test)))
#
# polynimial_svc = SVC(kernel='poly')
# polynimial_svc.fit(X_train,y_train)
#
# print('Polynomial 학습 데이터 점수 : {}:'.format(polynimial_svc.score(X_train,y_train)))
# print('Polynomial평가 데이터 점수 : {}:'.format(polynimial_svc.score(X_test,y_test)))
#
# rbf_svc = SVC(kernel='rbf')
# rbf_svc.fit(X_train,y_train)
#
# print('RBF 학습 데이터 점수 : {}:'.format(rbf_svc.score(X_train,y_train)))
# print('RBF 평가 데이터 점수 : {}:'.format(rbf_svc.score(X_test,y_test)))
#
#
# # todo 매개변수 튜닝
#
# X,y = load_breast_cancer(return_X_y=True)
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)
#
# polynimial_svc = SVC(kernel='poly',degree=2,C=0.1,gamma='auto')# poly 는 degree 조절 가능
# polynimial_svc.fit(X_train,y_train)
#
# print('kerner=poly, degree{}, gamma{}'.format(2,0.1,'auto'))
# print('Polynomial 학습 데이터 점수 : {}:'.format(polynimial_svc.score(X_train,y_train)))
# print('Polynomial평가 데이터 점수 : {}:'.format(polynimial_svc.score(X_test,y_test)))
#
#
# rbf_svc = SVC(kernel='rbf',C=2.0,gamma='scale')
# rbf_svc.fit(X_train,y_train)
# print('kerner=poly, C{}, gamma{}'.format(2.0,'scale'))
# print('RBF 학습 데이터 점수 : {}:'.format(rbf_svc.score(X_train,y_train)))
# print('RBF 평가 데이터 점수 : {}:'.format(rbf_svc.score(X_test,y_test)))

# todo 데이터 전처리 svm 은 전처리 해야 좋은 결과
X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

model = SVC()
model.fit(X_train,y_train)

print('SVC 학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('SVC 평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
model.fit(X_train,y_train)

print('SVC 학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('SVC 평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
model.fit(X_train,y_train)

print('minmax SVC 학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('minmax SVC 평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

# TODO linear svr

X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVR('linear')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

# X_comp = TSNE(n_components=1).fit_transform(X)
# plt.scatter(X_comp,y)
#
# model.fit(X_comp,y)
# predict = model.predict(X_comp)
# plt.scatter(X_comp,y)
# plt.scatter(X_comp,predict,color ='r')

estimator = make_pipeline(StandardScaler(),SVR(kernel='linear'))
# cross_validate(
#     estimator = estimator,
#     X=X , y=y,
#     cv=5,
#     n_jobs = multiprocessing.cpu_count(),
#     verbose = True
# )

pipe = Pipeline([('scaler',StandardScaler()),
                 ('model',SVR(kernel='linear'))])

param_grid = [{'model__gamma': ['scale','auto'],
               'model__C': [1.0,0.1,0.01],
               'model__epsilon': [1.0,0.1,0.01]}]

gs = GridSearchCV(
    estimator = pipe,
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    cv=5,
    verbose=True
)

print(gs.fit(X,y))
print(gs.best_estimator_)

# TODO LINEAR svc
X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

def make_meshgrid(x,y,h=.02):
    x_min,x_max= x.min()-1, x.max() +1
    y_min,y_max = y.min()-1,y.max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    return xx,yy

def plot_contours(clf,xx,yy,**params):
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z= Z.reshape(xx.shape)
    out = plt.contourf(xx,yy,Z,**params)

    return out
#
# X_comp = TSNE(n_components=2).fit_transform(X)
# X0,X1 = X_comp[:,0],X_comp[:,1]
# xx,yy = make_meshgrid(X0,X1)
#
# model.fit(X_comp,y)
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
# plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()



X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))
#
#
# X_comp = TSNE(n_components=2).fit_transform(X)
# X0,X1 = X_comp[:,0],X_comp[:,1]
# xx,yy = make_meshgrid(X0,X1)
#
# model.fit(X_comp,y)
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
# plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()



X,y = load_wine(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

def make_meshgrid(x,y,h=.02):
    x_min,x_max= x.min()-1, x.max() +1
    y_min,y_max = y.min()-1,y.max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    return xx,yy

def plot_contours(clf,xx,yy,**params):
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z= Z.reshape(xx.shape)
    out = plt.contourf(xx,yy,Z,**params)

    return out
#
# X_comp = TSNE(n_components=2).fit_transform(X)
# X0,X1 = X_comp[:,0],X_comp[:,1]
# xx,yy = make_meshgrid(X0,X1)
#
# model.fit(X_comp,y)
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
# plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

#todo kernel svc

X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

#
# X_comp = TSNE(n_components=2).fit_transform(X)
# X0,X1 = X_comp[:,0],X_comp[:,1]
# xx,yy = make_meshgrid(X0,X1)
#
# model.fit(X_comp,y)
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
# plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

#붓꽃
X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))

#
# X_comp = TSNE(n_components=2).fit_transform(X)
# X0,X1 = X_comp[:,0],X_comp[:,1]
# xx,yy = make_meshgrid(X0,X1)
#
# model.fit(X_comp,y)
# plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
# plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
# plt.show()

X,y = load_wine(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf')
model.fit(X_train,y_train)

print('학습 데이터 점수 : {}:'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}:'.format(model.score(X_test,y_test)))


X_comp = TSNE(n_components=2).fit_transform(X)
X0,X1 = X_comp[:,0],X_comp[:,1]
xx,yy = make_meshgrid(X0,X1)

model.fit(X_comp,y)
plot_contours(model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.7)
plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.show()