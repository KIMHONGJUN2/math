import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

df_adv = pd.read_csv('adv.csv',index_col=0)
print(df_adv.shape)

plt.scatter(df_adv.TV,df_adv.sales)
plt.title('Scatter plot')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()

X = df_adv.loc[:,['TV']]
Y = df_adv['sales']
print(X.shape)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.3, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)
regr.score(X_train,Y_train)

print(regr.coef_)

Y_pred = regr.predict(X_test)
np.mean((Y_pred - Y_test)**2)

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,Y_pred,color='blue',linewidth=3)
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()

df_credit = pd.read_csv('creditset.csv',index_col=0)
print(df_credit.shape)

X= df_credit.loc[:,['income','age','loan']]
Y = df_credit['default10yr']
print(X.shape)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.3, random_state=42)
model = linear_model.LogisticRegression()
model.fit(X,Y)
print(model.coef_)

#TSET 데이터로  모형의 성능 중 정분율 구하기
Y_pred = model.predict(X_test)
Y_pred2 = [0 if x < 0.5 else 1 for x  in Y_pred]
Y_pred3 = Y_pred2 == Y_test
np.mean(Y_pred3 == Y_test)

confusion_matrix(Y_test,Y_pred3)
print(classification_report(Y_test,Y_pred3))