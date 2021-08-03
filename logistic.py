# 로지스틱 회귀
# 클래스가 2개인 이진 분류를 위한 모델
# 선형 회귀모델 시그모이드 함수를 적용
# 목적함수를 최소화 하는 파라미터 w를 찾는것

#todo 예제
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

from sklearn.datasets import make_classification
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

samples = 1000
X,y = make_classification(n_samples=samples,n_features=2,
                          n_informative=2,n_redundant=0,
                          n_clusters_per_class=1)

fig,ax = plt.subplots(1,1,figsize=(10,6))

ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('y')

for i in range(samples):
    if y[i] ==0:
        ax.scatter(X[i,0],X[i,1],alpha=0.5,edgecolors ='k',marker='^',color='r')
    else:
        ax.scatter(X[i,0],X[i,1],edgecolors='k', alpha=0.5,marker='v',color='b')
#plt.show()

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
print('학습 데이터 점수 : {}'.format(model.score(X_train,y_train)))
print('평가 데이터 점수 : {}'.format(model.score(X_test,y_test)))

scores = cross_val_score(model,X,y,scoring='accuracy',cv=10)
print('CV 평균 점수 : {}'.format(scores.mean()))
print(model.intercept_,model.coef_)

x_min, x_max = X[:,0].min() - .5,X[:,0].max()+.5
y_min, y_max = X[:,0].min() - .5,X[:,0].max()+.5
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
Z = model.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(10,6))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X[:,0],X[:,1],c=np.abs(y-1),edgecolors='k',alpha=0.5,cmap=plt.cm.coolwarm)
plt.xlabel('X')
plt.xlabel('Y')

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.xticks()
plt.yticks()

plt.show()