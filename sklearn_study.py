import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# x = 10 * np.random.randn(50)
# y = 2 * x+np.random.randn(50)
# plt.scatter(x,y)
# #plt.show()
#
#
# # TODO
# # 1. 적절한 estimator 클래스를 임포트해서 모델의 클래스 선택
# from sklearn.linear_model import LinearRegression
#
# # TODO 2
# # 2. 클래스를 원하는 값으로 인스턴스화해서 모델의 하이파라미터 선택
# model = LinearRegression(fit_intercept=True)
# print(model) #j-jobs - 여러 코어 사용해 병렬로 처리
#
# # TODO 3
# # 3. 데이터를 특징 배열과 대상 벡터로 배치
# X = x[:, np.newaxis]
#
# # TODO 4
# # 4. 모델 인스턴스의 FIT() 메서드를 호출해 모델을 데이터에 적합
# model.fit(X,y)
#
# # TODO 5
# # 5. 모델을 새 데이터에 적용
# xfit = np.linspace(-30,31)
# Xfit = xfit[:, np.newaxis]
# yfit = model.predict(Xfit)
#
# plt.scatter(x,y)
# plt.plot(xfit,yfit,'--r')
# plt.show()


#----------------------------------------- dataset
from sklearn.datasets import load_diabetes

# diabetes = load_diabetes()
# print(diabetes)
# print(diabetes.data)
# print(diabetes.target)
# print(diabetes.DESCR ) #dataset 정보
#
# print(diabetes.feature_names)
# print(diabetes.data_filename)
# print(diabetes.target_filename)
#
# #train_test_split: 학습/테스트 데이터 세트 분리
#
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn.datasets import  load_diabetes
#
# diabetes = load_diabetes()
# X_train,X_test,y_train,y_test = train_test_split(diabetes.data,diabetes.target,test_size=0.3)
#
# model = LinearRegression()
# model.fit(X_train,y_train)
#
# print('학습데이터 점수 : {}'.format(model.score(X_train,y_train)))
# print('학습데이터 점수 : {}'.format(model.score(X_test,y_test)))
#
# predicted = model.predict(X_test)
# expected = y_test
# plt.figure(figsize=(8,4))
# plt.scatter(expected,predicted)
# plt.plot([30,350],[30,350],'--r')
# plt.tight_layout()
# #plt.show()
#
from sklearn.model_selection import cross_val_score,cross_validate
#
# scores = cross_val_score(model,diabetes.data,diabetes.target,cv=5)
# print('교차검증 정확도 {}'.format(scores))
# print('교차검증 정확도 {} +/-{}'.format(np.mean(scores),np.std(scores)))
#
# #교차검증과 최적 하이퍼 파리미터 찾기
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pandas as pd
#
# alpha =[0.001,0.01,0.1,1,10,100,1000]
# param_grid = dict(alpha=alpha)
#
# gs = GridSearchCV(estimator=Ridge(),param_grid=param_grid,cv=10)
# result = gs.fit(diabetes.data,diabetes.target)
#
# print('최적 점수 : {}'.format(result.best_score_))
# print('최적 파라미터 : {}'.format(result.best_params_))
# print(gs.best_estimator_)
# print(pd.DataFrame(result.cv_results_))

# ------------------------------ multiprocessing

import multiprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# iris = load_iris()
#
# param_grid = [ {'penalty': ['l1','l2'],
#                 'C':[1.5,2.0,2.5,3.0,3.5]}]
#
# gs = GridSearchCV(estimator=LogisticRegression(),param_grid=param_grid,
#                   scoring='accuracy',cv=10,n_jobs=multiprocessing.cpu_count())
# result = gs.fit(iris.data,iris.target)
#
# print('최적 점수 : {}'.format(result.best_score_))
# print('최적 파라미터 : {}'.format(result.best_params_))
# print(gs.best_estimator_)
# print(pd.DataFrame(result.cv_results_))

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
print(iris_df.describe())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print(iris_df_scaled.describe())

X_train,X_test,y_train,y_test = train_test_split(iris_df_scaled,iris.target,test_size=0.3)
model = LogisticRegression()
model.fit(X_train,y_train)

print('학습데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가데이터 점수 : {}'.format(model.score(X_test,y_test)))

# minmaxscaler 정규화 클래스
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
iris_scaled = scaler.fit_transform(iris_df)
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print(iris_df_scaled.describe())

X_train,X_test,y_train,y_test = train_test_split(iris_df_scaled,iris.target,test_size=0.3)
model = LogisticRegression()
model.fit(X_train,y_train)

print('학습데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가데이터 점수 : {}'.format(model.score(X_test,y_test)))

# 성능 평가 지표
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X,y = make_classification(n_samples=1000,n_features=2,n_informative=2,
                          n_redundant=0,n_clusters_per_class=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = LogisticRegression()
model.fit(X_train,y_train)

print('훈련 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

predict = model.predict(X_test)
print('정확도 : {}'.format(accuracy_score(y_test, predict)))
# 정확도만 보고 완성도를 판단하면 안된다.

# 오차행렬
#true negative : 예측값을 negative 값 0으로 예측 했고 실제도 negative 0
#false positive : 예측값을 positive 1 로 예측 , 실제값은 negative 0
#flase negative : 예측값을 negative 0 예측, 실제값은 positive 1
#true positive : 에측값을 positive 1 , 실제값도 positive 1

from sklearn.metrics import confusion_matrix

comfat = confusion_matrix(y_true=y_test,y_pred=predict)
print(comfat)

fig,ax = plt.subplots(figsize  = (2.5,2.5))
ax.matshow(comfat,cmap=plt.cm.Blues,alpha = 0.3)
for i in range(comfat.shape[0]):
    for j in range(comfat.shape[1]):
        ax.text(x=j,y=i,s=comfat[i,j],va='center',ha ='center')

# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.tight_layout
# plt.show()

# TODO
# 정밀도 = TP / (FP+TP)
# 재현율 = TP / (FN+TP)
# 정확도 = (TN+TP) / (TN+FP+FN+TP)
# 오류율 = (FN+FP) / (TN +FP +FN +TP)
from sklearn.metrics import precision_score,recall_score

precision = precision_score(y_test,predict)
recall = recall_score(y_test,predict)

print('정밀도 : {}'.format(precision))
print('재현율 : {}'.format(recall))

# f1 스코어 정밀도와 재현율 결합 지표
from sklearn.metrics import f1_score

f1 = f1_score(y_test,predict)
print('f1 score {}'.format(f1))

# roc 곡선과 auc
#fpr 이 변할 때 tpr 이 어떻게 변하는지 나타내는 곡선
#auc 값은 roc 곡선 밑에 면적을 구한 값(1에 가까울 수록 좋음)

from sklearn.metrics import roc_curve

pred_proba_class1 = model.predict_proba(X_test)[:,1]
fprs,tprs, thresholds = roc_curve(y_test,pred_proba_class1)

plt.plot(fprs,tprs,label = 'ROC')
plt.plot([0,1],[0,1],'--k',label = 'Random')
start ,end = plt.xlim()
plt.xticks(np.round(np.arange(start,end,0.1),2))
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('FPR(1-Sensitivity')
plt.ylabel('TPR(Recall')
plt.legend();
plt.show()

from sklearn.metrics import roc_auc_score

roc_acu = roc_auc_score(y_test,predict)

print('roc auc score: {}'.format(roc_acu))