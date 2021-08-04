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
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.datasets import load_boston,fetch_california_housing
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline

# todo k 최근접 이웃 분류
# 입력 데이터 포인트와 가장 가까운 k 개의 훈련 데이터 포인트가 출력
# k개의 데이터 포인트 중 가장 많은 클래스가 예측 결과

