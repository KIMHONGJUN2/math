# todo 앙상블 (ensemble)
# 일반화와 강건성 을 향상시키기 위해 여러 모델의 예측값을 결합
# 평균 방법과 부스팅 방법 2가지 종류
# 평균방법
# -- 여러개의 추정값을 독립적으로 구한뒤 평균을 구함
# -- 결합 추정값은 분산이 줄어들기 떄문에 단일 추정값보다 좋은 성능을 보임
#부스팅 방법
# -- 순차적으로 모델 생성
# -- 결합된 모델의 편향 감소 위해 노력
# -- 부스팅 방법 목표는 여러개의 약한 모델 결합해 하나의 강한 모델 구축
#bagging mate estimator
# -- 원래 훈련 데이터셋의 일부로 여러 모델 훈련
# -- 분산을 줄이고 과적합 막음
# -- 강력하고 복잡한 모델에서 잘 동작
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from sklearn.datasets import load_boston,load_diabetes
from  sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

# todo 아이리스 데이터 사용---------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 와인 데이터 사용시-----------------------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 유방암 데이터 사용시-------------------------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))


# todo svc @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# todo 아이리스 데이터 사용시 ------
# todo 아이리스 데이터 사용---------------------
base_model = make_pipeline(
    StandardScaler(),
    SVC()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))


# todo wine 데이터 사용시--------------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#todo 유방암 데이터 사용시 ----------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))


# todo decision tree @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
base_model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo wine 데이터 사용시--------------------------
base_model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#todo 유방암 데이터 사용시 ----------------------
base_model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier()
)
bagging_model = BaggingClassifier(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))


# TODO bagging 을 사용한 회귀 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
diabetes = load_diabetes()
boston = load_boston()

# todo 보스턴 데이터 사용시 -------------------
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 당뇨병 데이터 사용시 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo svr 사용시 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# todo 보스턴 데이터 사용시 -------------------
base_model = make_pipeline(
    StandardScaler(),
    SVR()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 당뇨병 데이터 사용시 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
base_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# TODO DeCISION TREE 사용시@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# todo 보스턴 데이터 사용시 -------------------
base_model = make_pipeline(
    StandardScaler(),
    SVR()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 당뇨병 데이터 사용시 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
base_model = make_pipeline(
    StandardScaler(),
    DecisionTreeRegressor()
)
bagging_model = BaggingRegressor(base_model,n_estimators=10,max_samples=0.5,max_features=0.5)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
#todo bagging 모델 사용시
cross_val = cross_validate(
    estimator=bagging_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo forests of randomized trees
# -- 두개의 평균화 알고리즘
# -- random forest , extra trees
# 모델 구성에 임의성을 추가해 다양한 모델 집합 생성
# 각 모델의 평균
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

#TODO 랜덤 포레스트 분류@@@@@@@@@@@@@@@@@@@@@@@@@@@
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
# todo 와인 데이터 사용시-----------------------------------

# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 유방암 데이터 사용시-------------------------------------
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# TODO  랜덤 포레스트 회귀 @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# TODO 보스턴 데이터 ---------------------
base_model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 당뇨병 데이터 사용시 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=base_model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#TODO Extremely randomized 분류 @@@@@@@@@@@@@@@@@@@@@@@@@@@
model = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier()
)
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
# todo 와인 데이터 사용시-----------------------------------

# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 유방암 데이터 사용시-------------------------------------
# todo base 모델 사용시
cross_val = cross_validate(
    estimator=model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo extremely Randomized trees 회귀@@@@@@@@@@@@@@@@@@@@@@@
model  = make_pipeline(
    StandardScaler(),
    ExtraTreesRegressor()
)

# todo 보스턴 데이터 사용시 ---------
cross_val = cross_validate(
    estimator=model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 유방암 데이터 사용시 ---------
cross_val = cross_validate(
    estimator=model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo random forest, extra tree 시각화@@@@@@@@@@@@@@@@@@2
import  numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from matplotlib.colors import  ListedColormap
from sklearn.tree import DecisionTreeClassifier

n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02
plot_step_coarser = 0.5
RANDOM_SEED = 13

iris = load_iris()
plot_idx = 1
models = [DecisionTreeClassifier(max_depth=None),
         RandomForestClassifier(n_estimators=n_estimators),
         ExtraTreesClassifier(n_estimators=n_estimators)]
plt.figure(figsize=(16,8))


# for pair in ([0,1],[0,2],[2,3]):
#     for model in models:
#         X = iris.data[:,pair]
#         y = iris.target
#
#         idx = np.arange(X.shape[0])
#         np.random.seed(RANDOM_SEED)
#         np.random.shuffle(idx)
#         X = X[idx]
#         y = y[idx]
#
#         mean = X.mean(axis = 0)
#         std = X.std(axis =0)
#         X = (X -mean)/std
#
#         model.fit(X,y)
#
#         model_title = str (type(model)).split('.')[-1][:-2][:-len('Classifier')]
#
#         plt.subplot(3,3,plot_idx)
#         if plot_idx <= len(models):
#             plt.title(model_title,fontsize = 9)
#
#         x_min,x_max = X[:,0].min() -1,X[:,0].max() +1
#         y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
#         xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
#                             np.arange(y_min,y_max,plot_step))
#
#         if isinstance(model,DecisionTreeClassifier):
#             Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             cs = plt.contourf(xx,yy,Z,cmap=cmap)
#         else:
#             estimator_alpha = 1.0 /len(model.estimators_)
#             for tree in model.estimators_:
#                 Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
#                 Z=Z.reshape(xx.shape)
#                 cs = plt.contourf(xx,yy,Z,alpha = estimator_alpha,cmap=cmap)
#
#             xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min,x_max,plot_step_coarser),
#                                                  np.arange(y_min,y_max,plot_step_coarser))
#             Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
#                                              yy_coarser.ravel()]).reshape(xx_coarser.shape)
#             cs_points = plt.scatter(xx_coarser,yy_coarser,s=15,
#                                     c=Z_points_coarser,cmap=cmap,
#                                     edgecolors='none')
#
#             plt.scatter(X[:,0],X[:,1],c=y,
#                         cmap=ListedColormap(['r','y','b']),
#                         edgecolors='k',s=20)
#             plot_idx +=1
#
#         plt.suptitle('Classifiers',fontsize = 12)
#         plt.axis('tight')
#         plt.tight_layout(h_pad=0.2,w_pad=0.2
#                          ,pad=2.5)
#
#         plt.show()

#todo adaboost
# -- 대표적인 부스팅 알고리즘
# -- 일련의 약한 모델들을 학습
# -- 수정된 버전의 데이터를 반복 학습 ( 가중치가 적용된)
# -- 첫 단계에서 원본 데이터를 학습하고 연속적인 반복마다 개별 샘플에 대한
#가중치가 수정되고 다시 모델이 학습
# 잘못 예측된 샘플은 가중치 증가 , 올바르게 예측된 샘플은 가중치 감소
# 각각의 약한 모델들은 예측하기 어려운 샘플에 집중하게 됨

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

model= make_pipeline(
    StandardScaler(),
    AdaBoostClassifier()
)
cross_val = cross_validate(
    estimator=model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

cross_val = cross_validate(
    estimator=model,
    X=wine.data, y =wine.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))
cross_val = cross_validate(
    estimator=model,
    X=cancer.data, y =cancer.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#todo 회귀 모델
model = make_pipeline(
    StandardScaler(),
    AdaBoostRegressor()
)
cross_val = cross_validate(
    estimator=model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

cross_val = cross_validate(
    estimator=model,
    X=diabetes.data, y =diabetes.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#todo gradient tree boosting
# -- 임의의 차별화 가능한 손실함수로 일반화한 부스팅 알고리즘
# -- 웹검색, 분류 및 회귀 등 다양한 분야 모두 사용

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

model = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier()
)
cross_val = cross_validate(
    estimator=model,
    X=iris.data, y =iris.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

# todo 회귀 @@@@@@@@@@@22
model = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor()
)
cross_val = cross_validate(
    estimator=model,
    X=boston.data, y =boston.target,
    cv=5
)
print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))

#todo 투표 기반 분류
# -- 서로 다른 모델들의 결과를 투표를 통해 결합
# -- 두가지 방법으로 투표 가능
# -- 가장 많이 예측된 클래스를 정답
# -- 예측된 확률 가중치 평균

from  sklearn.svm import SVC
from sklearn.naive_bayes import  GaussianNB
from  sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

model1 =SVC()
model2 = GaussianNB()
model3 = RandomForestClassifier()
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3)],
    voting='hard'
)
for model in (model1,model2,model3,vote_model):
    model_name = str(type(model)).split('.')[-1][:-2]
    scores = cross_val_score(model,iris.data,iris.target,cv=5)
    print('Accuracy : %0.2f (+/- %0.2f) [%s]' % (scores.mean(),scores.std(),model_name))


# todo soft
model1 =SVC(probability=True)
model2 = GaussianNB()
model3 = RandomForestClassifier()
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3)],
    voting='soft'
    ,weights=[2,1,2]
)
for model in (model1,model2,model3,vote_model):
    model_name = str(type(model)).split('.')[-1][:-2]
    scores = cross_val_score(model,iris.data,iris.target,cv=5)
    print('Accuracy : %0.2f (+/- %0.2f) [%s]' % (scores.mean(),scores.std(),model_name))

#todo 결정 경계 시각화
from sklearn.tree import DecisionTreeClassifier
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from itertools import product
from pipes import makepipeline

X = iris.data[:,[0,2]]  # 4개의 피처 중 2개 사용
y = iris.target

model1 = DecisionTreeClassifier(max_depth=4)
model2=KNeighborsClassifier(n_neighbors=7)
model3 = SVC(gamma=1,kernel='rbf',probability=True)
vote_model = VotingClassifier(estimators=[('dt',model1),('knn',model2),('svc',model3)],
                              voting='soft',weights=[2,1,2])
model1 = model1.fit(X,y)
model2 = model2.fit(X,y)
model3 = model3.fit(X,y)
vote_model = vote_model.fit(X,y)

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

# f, axarr = plt.subplots(2,2,sharex='col',sharey='row',figsize=(12,8))
# for idx,model, tt in zip (product([0,1],[0,1]),
#                          [model1,model2,model3,vote_model],
#                           ['Decision Tree(depth=4)','KNN(k=7)',
#                            'Kernel SVM','Soft Voting']):
#     Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha = 0.4)
#     axarr[idx[0],idx[1]].scatter(X[:,0],X[:,1],c=y,s=20,edgecolor = 'k')
#     axarr[idx[0],idx[1]].set_title(tt)
# plt.show()

#todo 투표 기반 회귀
# -- 서로 다른 모델의 예측 값의 평균을 사용
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
#
# model1 = LinearRegression()
# model2 = GradientBoostingRegressor()
# model3 = RandomForestRegressor()
# vote_model = VotingRegressor(
#     estimators=[('linear',model1),('gbr',model2),('rfr',model3)],
#     weights=[1,1,1]
# )
#
# for model in (model1,model2,model3,vote_model):
#     model_name = str(type(model)).split('.')[-1][:-2]
#     scores = cross_val_score(model,boston.data,boston.target,cv=5)
#     print('R2: %0.2f (+/- %0.2f) [%s]' % (scores.mean(),scores.std(),model_name))
#
# # todo 회귀식 시각화
# X = boston.data[:,0].reshape(-1,1)
# y = boston.target
#
# model1 = model1.fit(X,y)
# model2 = model2.fit(X,y)
# model3 = model3.fit(X,y)
# vote_model = vote_model.fit(X,y)
#
# x_min ,x_max = X.min()-1,X.max()+1
# xx = np.arange(x_min-1,x_max+1,0.1)
#
# f,axarr = plt.subplots(2,2,sharex='col',sharey='row',figsize=(12,8))

# for idx,model ,tt in zip(product([0,1],[0,1]),
#                          [model1,model2,model3,vote_model],
#                          ['Linear Regression','Gradient Boosting','Random Forest','Voting']):
#
#     Z = model.predict(xx.reshape(-1,1))
#
#     axarr[idx[0],idx[1]].plot(xx,Z,c='r')
#     axarr[idx[0], idx[1]].scatter(X,y,s=20,edgecolor = 'k')
#     axarr[idx[0], idx[1]].set_title(tt)
#
# plt.show()

#todo 스택 일반화
# -- 각 모델의 예측값을 최종 모델의 입력으로 사용
# -- 모델의 편향을 줄이는데 효과적
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

# estimators = [('ridge',Ridge()),
#               ('lasso',Lasso()),
#               ('svr',SVR())]
#
# reg = make_pipeline(
#     StandardScaler(),
#     StackingRegressor(
#         estimators=estimators,
#         final_estimator=GradientBoostingRegressor()
#     )
# )
# print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
# print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
# print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))


# todo 시각화
# X = boston.data[:,0].reshape(-1,1)
# y = boston.target
#
# model1 = Ridge()
# model2 = Lasso()
# model3 = SVR()
# reg = StackingRegressor(
#     estimators=estimators,
#     final_estimator=GradientBoostingRegressor()
# )
#
# model1 = model1.fit(X,y)
# model2 = model2.fit(X,y)
# model3 = model3.fit(X,y)
# reg =reg.fit(X,y)
#
# x_min ,x_max = X.min()-1,X.max()+1
# xx = np.arange(x_min-1,x_max+1,0.1)
#
# f,axarr = plt.subplots(2,2,sharex='col',sharey='row',figsize=(12,8))

# for idx,model ,tt in zip(product([0,1],[0,1]),
#                          [model1,model2,model3,vote_model],
#                          ['Ridge','Lasso','SVM','Stack']):
#
#     Z = model.predict(xx.reshape(-1,1))
#
#     axarr[idx[0],idx[1]].plot(xx,Z,c='r')
#     axarr[idx[0], idx[1]].scatter(X,y,s=20,edgecolor = 'k')
#     axarr[idx[0], idx[1]].set_title(tt)
#
# plt.show()


#todo 스택 분류
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

estimators = [('logistic',LogisticRegression(max_iter=10000)),
              ('svc',SVC()),
              ('naive',GaussianNB())]
# clf = StackingClassifier(
#     estimators=estimators,
#     final_estimator=RandomForestClassifier()
# )
#
# cross_val = cross_validate(
#     estimator=clf,
#     X=iris.data, y =iris.target,
#     cv=5
# )
# print('avg fit time: {}(+/- {})'.format(cross_val['fit_time'].mean(),cross_val['fit_time'].std()))
# print('avg score time: {}(+/- {})'.format(cross_val['score_time'].mean(),cross_val['score_time'].std()))
# print('avg test score: {}(+/- {})'.format(cross_val['test_score'].mean(),cross_val['test_score'].std()))



# todo 시각화
X = iris.data[:,[0,2]]  # 4개의 피처 중 2개 사용
y = iris.target

model1 = LogisticRegression(max_iter=10000)
model2=SVC()
model3 = GaussianNB()
stack = StackingClassifier(
    estimators =estimators,
    final_estimator=RandomForestClassifier()
)
model1 = model1.fit(X,y)
model2 = model2.fit(X,y)
model3 = model3.fit(X,y)
stack = stack.fit(X,y)

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

f, axarr = plt.subplots(2,2,sharex='col',sharey='row',figsize=(12,8))
for idx,model, tt in zip (product([0,1],[0,1]),
                         [model1,model2,model3,vote_model],
                          ['logistic','svc',
                           'gau','stack']):
    Z = model.predict(xx.reshape(-1, 1))

    axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha = 0.4)
    axarr[idx[0],idx[1]].scatter(X[:,0],X[:,1],c=y,s=20,edgecolor = 'k')
    axarr[idx[0],idx[1]].set_title(tt)
plt.show()