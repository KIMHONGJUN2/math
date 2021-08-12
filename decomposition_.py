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

#todo sparse pca
# pca의 주요 단점 중 하나는 주성분들이 보통 모든 입력 변수들의 션형결합으로 나타난다는 것
# 희소 주성분분석은 몇개 변수들만의 선형결합으로 주성분을 나타냄으로써 극복

model = SparsePCA(n_components=2,random_state=0)
model.fit(iris)
transformed_iris = model.transform(iris)
# plot_iris(transformed_iris,labels)

model = SparsePCA(n_components=2,random_state=0)
model.fit(faces)
faces_component = model.components_
# show_faces(faces_component)

#todo truncated singular value decomposition(truncated svd)
# pca는 정방행렬에 대해서만 행렬 분해 가능
# svd는 행과 열이 다른 행렬도 분해 가능
# 희소 행렬도 가능

model = TruncatedSVD(n_components=2,random_state=0)
model.fit(iris)
transformed_iris = model.transform(iris)
# plot_iris(transformed_iris)

model = TruncatedSVD(n_components=2,random_state=0)
model.fit(faces)
faces_component = model.components_
# show_faces(faces_component)

#todo dictionary learning
# sparse code 를 사용하여 데이터를 가장 잘 나타내는 사전 찾기
# sparse coding은 기저벡터를 기반으로 데이터를 효율적으로 표현하기 위해개발
model = DictionaryLearning(n_components=2,random_state=0)
model.fit(iris)
transformed_iris = model.transform(iris)
# plot_iris(transformed_iris,labels)

model = DictionaryLearning(n_components=2*3,random_state=0)
model.fit(faces)
faces_component = model.components_
# show_faces(faces_component)


#todo factor analysis
# 요인분석은 변수들 간의 상관관계를 고려해 내재된 개념인 요인들을 추출해내는 방법
# 유사한 변수들끼리 묶어주는 방법
# pca는 오차를 고려하지 않고 요인 분석은 오차를 고려함

# model = FactorAnalysis(n_components=2,random_state=0)
# model.fit(iris)
# transformed_iris = model.transform(iris)
# # plot_iris(transformed_iris)
#
# model = FactorAnalysis(n_components=2*3,random_state=0)
# model.fit(faces)
# faces_component = model.components_

#todo independaent component analysis(ICA)
# 다변량의 신호를 통계적으로 독립적인 하부 성분으로 분리하는 방법
# model =  FastICA(n_components=2,random_state=0)
# model.fit(iris)
# transformed_iris = model.transform(iris)
# # plot_iris(transformed_iris)
#
# model = FastICA(n_components=2*3,random_state=0)
# model.fit(faces)
# faces_component = model.components_
# show_faces(faces_component)

#todo non-negative matrix factorization
# 음수 미포함 행렬분해는 음수를 포함하지 않은 행렬 v를 음수를 포함하지 않은 행렬 w 와 h 의 곱으로 분해하는 알고리즘
# model = NMF(n_components=2,random_state=0)
# model.fit(iris,labels)
# transformed_iris = model.transform(iris)
#
# model = NMF(n_components=2*3,random_state=0)
# model.fit(faces)
# faces_component = model.components_

#todo latent dirichlet allocation(LDA)
# 잠재 디리클레 할당은 이산 자료들에 대한 확률적 생성 모형
# 디리클레 분포에 따라 잠재적인 의미 구조를 파악

# model = LatentDirichletAllocation(n_components=2,random_state=0)
# model.fit(iris,labels)
# transformed_iris = model.transform(iris)
# plot_iris(transformed_iris)

# model = LatentDirichletAllocation(n_components=2*3,random_state=0)
# model.fit(faces)
# faces_component = model.components_
# show_faces(faces_component)

#todo linear didcriminant analysis (LDA)
# 데이터세트를 저차원 공간에 투영해 차원을 축소
# 지도학습 분류에서 사용
# model = LinearDiscriminantAnalysis(n_components=2)
# model.fit(iris,labels)
# transformed_iris = model.transform(iris)
# plot_iris(transformed_iris,labels)

#압축된 표현을 사용한 학습
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import SVC
from  sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score,train_test_split

def min_max_scale(x):
    min_value , max_value = np.min(x,0),np.max(x,0)
    x = (x-min_value)/(max_value-min_value)
    return x

def plot_digits(digits,labels):
    digits = min_max_scale(digits)
    ax = plt.subplot(111,projection ='3d')
    for i in range(digits.shape[0]):
        ax.text(digits[i,0],digits[i,1],digits[i,2],
                str(labels[i]),color = plt.cm.Set1(labels[i]/10.),
                    fontdict = {'weight': 'bold','size': 9})
    ax.view_init(4,-72)

digits = load_digits()
nmf = NMF(n_components=3)
nmf.fit(digits.data)
decomposed_digits = nmf.transform(digits.data)


print(digits.data.shape)
print(decomposed_digits.shape)
print(decomposed_digits)

plt.figure(figsize=(20,10))
plot_digits(decomposed_digits,digits.target)
plt.show()

#todo knn
knn = KNeighborsClassifier()
score = cross_val_score(
    estimator=knn,
    X=digits.data,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

score = cross_val_score(
    estimator=knn,
    X=decomposed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

#svm
svm = SVC()
score = cross_val_score(
    estimator=svm,
    X=decomposed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))


#decision tree
decision_tree = DecisionTreeClassifier()
score = cross_val_score(
    estimator=decision_tree,
    X=decomposed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

#random frest
random_forest = RandomForestClassifier()
score = cross_val_score(
    estimator=random_forest,
    X=decomposed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

#todo 복원된 표현을 사용한 학습
components = nmf.components_
reconstructed_digits = decomposed_digits @ components
plt.figure(figsize=(16,8))
plt.suptitle('Re-constructed digits')
for i in range (10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(reconstructed_digits[i].reshape(8,8))



score = cross_val_score(
    estimator=knn,
    X=reconstructed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

score = cross_val_score(
    estimator=svm,
    X=reconstructed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

score = cross_val_score(
    estimator=decision_tree,
    X=reconstructed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

score = cross_val_score(
    estimator=random_forest,
    X=reconstructed_digits,y=digits.target,
    cv=5
)
print('mean cross val score: {}(+/-{})'.format(score.mean(),score.std()))

#todo 이미지 복원
train_faces, test_faces = train_test_split(faces,test_size=0.1)
show_faces(train_faces) #학습 원본
show_faces(test_faces) #평가를 위한 원본

damaged_faces =[]
for face in test_faces:
    idx = np.random.choice(range(64*64),size=1024)
    damaged_face = face.copy()
    damaged_face[idx] = 0.
    damaged_faces.append(damaged_face)
show_faces(damaged_faces)

nmf =NMF(n_components=10)
nmf.fit(train_faces)

matrix1 = nmf.transform(damaged_faces)
matrix2 = nmf.components_
show_faces(matrix1 @ matrix2)

nmf = NMF(n_components=100)
nmf.fit(train_faces)

matrix1 = nmf.transform(damaged_faces)
matrix2 = nmf.components_
show_faces(matrix1 @ matrix2)