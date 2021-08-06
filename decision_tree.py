# todo 결정트리
#분류와 회긔에 사용되는 지도 학습 방법
#if-then-else 결정 규칙 통해 학습
# 장점
#1.이해,해석 쉽다
#2.시각화 용이
#3.많은 전처리 필요하지 않다
#4.수치형과 범주형 모두 다룰 수 있다.

import pandas as pd
import numpy as np
import graphviz
import multiprocessing
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_boston,load_diabetes
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
#
#붓꽃
iris = load_iris()

iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['target'] = iris.target
#print(iris_df)

#와인
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
wine_df['target'] = wine.target
#print(wine_df)

#유방암
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
cancer_df['target'] = cancer.target
#print(cancer_df)

#todo 회귀를 위한 데이터

#보스턴 주택 가격
boston = load_boston()
boston_df = pd.DataFrame(data=boston.data,columns=boston.feature_names)
boston['target'] = boston.target

#당뇨병 데이터
diabets = load_diabetes()
diabets_df = pd.DataFrame(data=diabets.data,columns=diabets.feature_names)
diabets_df['target'] = diabets.target

# todo 분류
#DecisionTreeClassifier는 분류를 위한 결정트리 모델
# 두개의 배열 x,y 를 입력
# x는 [n_sample,n_feature] 크기의 데이터 특성 배열
# y는 [n_sample]크기의 정답 배열
X = [[0,0],[1,1]]
y=[0,1]

model = tree.DecisionTreeClassifier()
model = model.fit(X,y)
print(model.predict([[2.,2.]]))
print(model.predict_proba([[2.,2.]]))

# todo 붓꽃 데이터 활용
#todo 전처리 없이 학습
model = DecisionTreeClassifier()
print(cross_val_score(
    estimator=model,
    X=iris.data,y=iris.target,
    cv=5,
    n_jobs=multiprocessing.cpu_count()
))
#todo 전처리 후 학습
model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier()
)
print(cross_val_score(
    estimator=model,
    X=iris.data,y=iris.target,
    cv=5,
    n_jobs=multiprocessing.cpu_count()
))   # 결정 트리는 규칙을 학습해서 전처리가 크게 영향을 미치지는 않는다.

# todo 학습된 결정트리 시각화
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

#todo 텍스트를 통한 시각화
r = tree.export_text(decision_tree=model,
                     feature_names=iris.feature_names)
print(r)

#todo plot tree 를 통한 시각화
#tree.plot_tree(model)

#todo graphviz 시각화
dot_data = tree.export_graphviz(decision_tree=model,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True,rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)

# 결정경계 시각화
n_classes =3
plot_color = 'ryb'
plot_step = 0.02

# plt.figure(figsize=(16,8))
#
# for pairidx,pair in enumerate([[0,1],[0,2],[0,3],
#                               [1,2],[1,3],[2,3]]):
#     X = iris.data[:,pair]
#     y= iris.target
#
#     model = DecisionTreeClassifier()
#     model = model.fit(X,y)
#
#     plt.subplot(2,3,pairidx + 1)
#
#     x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
#     y_min,y_max = X[:,1].max()-1,X[:,1].max()+1
#     xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
#                         np.arange(y_min,y_max,plot_step))
#     plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu)
#
#     plt.xlabel(iris.feature_names[pair[0]])
#     plt.ylabel(iris.feature_names[pair[1]])
#
#     for i ,color in zip(range(n_classes),plot_color):
#         idx = np.where(y==i)
#         plt.scatter(X[idx,0],X[idx,1],c=color,label= iris.target_names[i],
#                     cmap=plt.cm.RdYlBu,edgecolors='b',s=15)
#
# plt.suptitle('Decision surface')
# plt.legend(loc = 'lower right',borderpad=0,handletextpad = 0)
# plt.axis('tight')
# plt.show()

# todo 하이퍼파라미터를 변경해 보며 결정 경계 변화 확인

plt.figure(figsize=(16,8))

for pairidx,pair in enumerate([[0,1],[0,2],[0,3],
                              [1,2],[1,3],[2,3]]):
    X = iris.data[:,pair]
    y= iris.target

    model = DecisionTreeClassifier(max_depth=2)
    model = model.fit(X,y)

    plt.subplot(2,3,pairidx + 1)

    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
                        np.arange(y_min,y_max,plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    for i ,color in zip(range(n_classes),plot_color):
        idx = np.where(y==i)
        plt.scatter(X[idx,0],X[idx,1],c=color,label= iris.target_names[i],
                    cmap=plt.cm.RdYlBu,edgecolors='b',s=15)

plt.suptitle('Decision surface')
plt.legend(loc = 'lower right',borderpad=0,handletextpad = 0)
plt.axis('tight')
plt.show()

# todo wine 데이터 학습
# 전처리 없이 학습
model = DecisionTreeClassifier()
cross_val_score(
    estimator=model,
    X=wine.data,y=wine.target,
    cv=5,
    n_jobs=multiprocessing.cpu_count()
)
#전처리 후 학습
model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier()
)
cross_val_score(
    estimator=model,
    X=wine.data,y=wine.target,
    cv=5,
    n_jobs=multiprocessing.cpu_count()
)

# 학습된 결정 트리 시각화

# todo 회귀

#보스턴
model = DecisionTreeRegressor()
cross_val_score(
    estimator=model,
    X = boston.data, y = boston.target,
    cv=5,
    n_jobs=multiprocessing.cpu_count()
)
# 전처리 후 학습
model = make_pipeline(
    StandardScaler(),
    DecisionTreeRegressor()
)

# 학습된 결정 트리 시각화
model = DecisionTreeRegressor()
model.fit(boston.data,boston.target)
#텍스트를 통한 시각화
# 그래프비즈
dot_data = tree.export_graphviz(decision_tree=model,
                                feature_names=boston.feature_names
                            , filled=True,rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)

#회귀식 시각화
# plt.figure(figsize=(16,8))
# for pairidx, pair in enumerate([0,1,2]):
#     X = boston.data[:,pair].reshape(-1,1)
#     y = boston.target
#
#     model = DecisionTreeRegressor()
#     model.fit(X,y)
#
#     X_test = np.arange(min(X),max(X),0.1)[:,np.newaxis]
#     predict = model.predict(X_test)
#
#     plt.subplot(1,3,pairidx + 1)
#     plt.scatter(X,y,s =20,edgecolors='b',
#                 c = 'darkorange',label = 'data')
#     plt.plot(X_test,predict,color='skyblue', linewidth = 2)
#     plt.xlabel(boston.feature_names[pair])
#     plt.ylabel('Target')
#     plt.show()

# 하이퍼 파라미터 변경
# plt.figure(figsize=(16,8))
# for pairidx, pair in enumerate([0,1,2]):
#     X = boston.data[:,pair].reshape(-1,1)
#     y = boston.target
#
#     model = DecisionTreeRegressor(max_depth=3)
#     model.fit(X,y)
#
#     X_test = np.arange(min(X),max(X),0.1)[:,np.newaxis]
#     predict = model.predict(X_test)
#
#     plt.subplot(1,3,pairidx + 1)
#     plt.scatter(X,y,s =20,edgecolors='b',
#                 c = 'darkorange',label = 'data')
#     plt.plot(X_test,predict,color='skyblue', linewidth = 2)
#     plt.xlabel(boston.feature_names[pair])
#     plt.ylabel('Target')
#     plt.show()

