import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x = 10 * np.random.randn(50)
y = 2 * x+np.random.randn(50)
plt.scatter(x,y)
#plt.show()


# TODO
# 1. 적절한 estimator 클래스를 임포트해서 모델의 클래스 선택
from sklearn.linear_model import LinearRegression

# TODO 2
# 2. 클래스를 원하는 값으로 인스턴스화해서 모델의 하이파라미터 선택
model = LinearRegression(fit_intercept=True)
print(model) #j-jobs - 여러 코어 사용해 병렬로 처리

# TODO 3
# 3. 데이터를 특징 배열과 대상 벡터로 배치
X = x[:, np.newaxis]

# TODO 4
# 4. 모델 인스턴스의 FIT() 메서드를 호출해 모델을 데이터에 적합
model.fit(X,y)

# TODO 5
# 5. 모델을 새 데이터에 적용
xfit = np.linspace(-30,31)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x,y)
plt.plot(xfit,yfit,'--r')
plt.show()