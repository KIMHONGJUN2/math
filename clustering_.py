#todo 군집화
# 대표적인 비지도
# 레이블이 없음
import numpy as np
import matplotlib.pyplot as plt


from sklearn import cluster
from sklearn import  mixture
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#todo 데이터생성
def plot_data(dataset,position,title):
    X,y =dataset
    plt.subplot(position)
    plt.title(title)
    plt.scatter(X[:,0],X[:,1])

np.random.seed(0)
n_samples = 1500
random_state = 0
noise = 0.05

circles = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=noise,random_state=random_state)
moons = datasets.make_moons(n_samples=n_samples,noise=noise,random_state=random_state)
blobs = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
no_structure = np.random.rand(n_samples,2),None


# todo 시각화
# plt.figure(figsize=(12,12))
# plot_data(circles,221,'Circles')
# plot_data(moons,222,'Moons')
# plot_data(blobs,223,'Blobs')
# plot_data(no_structure,224,'no_structures')
#
# plt.show()

def fit_predict_plot(model,dataset,position,title):
    X,y = dataset
    model.fit(X)
    if hasattr(model,'label'):
        labels = model.labels_.astype(np.int)
    else:
        labels = model.predict(X)
    colors = np.array(['#30A9DE','#E53A40','#090707','#A593E0','#F6B352','#519D9E','#AA34B3'])
    ax = plt.subplot(position)
    ax.set_title(title)
    ax.scatter(X[:,0],X[:,1],color = colors[labels])



#todo k-평균 = k-means
# n개의 등분산 그룹으로 군집화
# 제곱합 함수 최소화
# 군집화 개수 지정

# fig = plt.figure(figsize=(12,12))
# fig.suptitle('k-Means')
#
# fit_predict_plot(cluster.KMeans(n_clusters=2,random_state=random_state),circles,221,'Circles')
# fit_predict_plot(cluster.KMeans(n_clusters=2,random_state=random_state),moons,222,'Moons')
# fit_predict_plot(cluster.KMeans(n_clusters=2,random_state=random_state),blobs,223,'Blobs')
# fit_predict_plot(cluster.KMeans(n_clusters=2,random_state=random_state),no_structure,224,'No str1ucture')
# plt.show()

# fig = plt.figure(figsize=(12,12))
# fig.suptitle('k-Means')
#
# fit_predict_plot(cluster.KMeans(n_clusters=4,random_state=random_state),circles,221,'Circles')
# fit_predict_plot(cluster.KMeans(n_clusters=4,random_state=random_state),moons,222,'Moons')
# fit_predict_plot(cluster.KMeans(n_clusters=4,random_state=random_state),blobs,223,'Blobs')
# fit_predict_plot(cluster.KMeans(n_clusters=4,random_state=random_state),no_structure,224,'No str1ucture')
# plt.show()

#todo 붓꽃 데이터 군집화
from sklearn.datasets import load_iris

iris = load_iris()

model = cluster.KMeans(n_clusters=3)
model.fit(iris.data)
predict = model.predict(iris.data)

idx  = np.where(predict==0)
iris.target[idx]

idx  = np.where(predict==1)
iris.target[idx]

idx  = np.where(predict==2)
iris.target[idx]

#todo 미니배치 k 평균

# fig = plt.figure(figsize=(12,12))
# fig.suptitle('MiniBatch k-Means')
#
# fit_predict_plot(cluster.MiniBatchKMeans(n_clusters=2,random_state=random_state),circles,221,'Circles')
# fit_predict_plot(cluster.MiniBatchKMeans(n_clusters=2,random_state=random_state),moons,222,'Moons')
# fit_predict_plot(cluster.MiniBatchKMeans(n_clusters=2,random_state=random_state),blobs,223,'Blobs')
# fit_predict_plot(cluster.MiniBatchKMeans(n_clusters=2,random_state=random_state),no_structure,224,'No str1ucture')
# plt.show()

#todo affinity propagation
# 샘플 쌍 끼리 메시지를 보내 군집 생성
# 샘플을 대표하는 적절한 예를 찾을때까지
# 군집의 개수를 자동으로

# fig = plt.figure(figsize=(12,12))
# fig.suptitle('Affinitu Propagation')
#
# fit_predict_plot(cluster.AffinityPropagation(damping=.9,preference=-200),circles,221,'Circles')
# fit_predict_plot(cluster.AffinityPropagation(damping=.9,preference=-200),moons,222,'Moons')
# fit_predict_plot(cluster.AffinityPropagation(damping=.9,preference=-200),blobs,223,'Blobs')
# fit_predict_plot(cluster.AffinityPropagation(damping=.9,preference=-200),no_structure,224,'No str1ucture')
# plt.show()






