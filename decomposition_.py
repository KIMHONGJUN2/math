#todo 분해 (decomposition)
# 큰 하나의 행렬을 여러개의 작은 행렬로 분해
# 분해 과정에서 중요한 정보만 남게됨

# 데이터 불러오기 및 시각화
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris,fetch_olivetti_faces
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,SparsePCA,TruncatedSVD,DictionaryLearning,FactorAnalysis,FastICA,NMF,LatentDirichletAllocation
from  sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

iris , labels = load_iris(return_X_y=True)
faces, _ = fetch_olivetti_faces(return_X_y=True,shuffle=True)
def plot_iris(iris,labels):
    plt.figure()
    colors = ['navy','purple','red']
    for xy,label in zip(iris,labels):
        plt.scatter(xy[0],xy[1],color=colors[label])

def show_faces(faces):
    plt.figure()
    num_rows,num_cols =2,3
    for i in range(num_rows*num_cols):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(np.reshape(faces[i],(64,64)),cmap=plt.cm.gray)
# plot_iris(iris[:,:2],labels)
# plt.show()
# show_faces(faces)
# plt.show()

#todo principal component analtsus(pca)
# pca를 사용해 iris 데이터 변환
# 150x4 크기의 데이터를 150x2 크기의 행렬로 압축
model = PCA(n_components=2,random_state=0)
model.fit(iris)
transformed_iris =model.transform(iris)
transformed_iris.shape

#plot_iris(transformed_iris,labels)

# faces pca
model = PCA(n_components=2*3,random_state=0)
model.fit(faces)
faces_component = model.components_
#show_faces(faces_component)

#todo incremental PCA
#pca는 svd 알고리즘 실행을 위해 전체 학습용 데이터 셋을 메모리에 올려야함
#incremental pca 학습 데이터를 미니 배치 단위로 나누어 사용
#학습 데이터가 크거나 온라인으로 pca적용이 필요할 때 유용
model  = IncrementalPCA(n_components=2)
model.fit(iris)
transformed_iris = model.transform(iris)
transformed_iris.shape

# plot_iris(transformed_iris,labels)

model = IncrementalPCA(n_components=2*3)
model.fit(faces)
faces_component = model.components_
faces_component.shape

# show_faces(faces_component)

#todo kernel pca
# 차원 축소를 위한 복잡한 비선형 투형
model = KernelPCA(n_components=2,kernel='rbf',random_state=0)
model.fit(iris)
transformed_iris = model.transform(iris)
#plot_iris(transformed_iris,labels)
#
# model = KernelPCA(n_components=2*3,kernel='rbf',random_state=0)
# model.fit(faces)
# faces_component = model.components_ #-------------------------- kernel에서는 컴포넌트 불가
# faces_component.shape

