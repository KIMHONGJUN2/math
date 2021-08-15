#todo 추천시스템 (recommender system)
# 두가지로 구분 가능
# 컨텐츠 기반 필터링 , 협업 필터링
# 두가지 조합한 hybrid 방식 가능
# 컨텐츠 = 지금까지 사용자의 이전 행동과 명시적 피드백을 통해 사용자가 좋아하는 것과 유사한 항목 추천
# 사용자와 항목간의 유사성을 동시에 사용해 추천

#todo surprise
from surprise import SVD
from  surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k',prompt=False)
data.raw_ratings[:10]
model = SVD()
cross_validate(model,data,measures=['rmse','mae'],cv=5,verbose=True)

#todo 컨텐츠 기반 필터링
# 사용자 이전 행동과 명시적 피드백을 통해 좋아하는 것과 유사한 항목 추천
# 나와 비슷한 취향의 사용자가 시청한 영화를 추천
# 유사도 기반
# 도메인 지식 필요
# 대상 쉽게 확장 가능

import numpy as np
from surprise import Dataset

data = Dataset.load_builtin('ml-100k',prompt=False)
raw_data = np.array(data.raw_ratings,dtype=int)

raw_data[:,0] -=1
raw_data[:,1] -=1

n_users = np.max(raw_data[:,0])   #유저 수
n_movies = np.max(raw_data[:,1])  #영화 수
shape = (n_users + 1,n_movies + 1)
print(shape)

adj_matrix = np.ndarray(shape,dtype=int)
for user_id,movie_id,rating,time in raw_data:
    adj_matrix[user_id][movie_id] =1.


my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id,best_match_vector =-1,-1,[]

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        similarity = np.dot(my_vector,user_vector)
        if similarity > best_match:
            best_match = similarity
            best_match_id = user_id
            best_match_vector = user_vector

print( 'Best Match: {}, Best Match ID: {}'.format(best_match,best_match_id))

recommend_list=[]
for i , log in enumerate(zip(my_vector,best_match_vector)):
    log1,log2 = log
    if log1 <1. and log2 > 0. :
        recommend_list.append(i)
print(recommend_list)

#todo 유클리드 거리를 사용한 추천

my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = 9999, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        euclidean = np.sqrt(np.sum(np.square(my_vector - user_vector)))
        if euclidean < best_match:
            best_match = euclidean
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))

recommend_list=[]
for i , log in enumerate(zip(my_vector,best_match_vector)):
    log1,log2 = log
    if log1 <1. and log2 > 0. :
        recommend_list.append(i)
print(recommend_list)

#todo 코사인 유사도 계산
def compute_cos_similarity(v1,v2):
    norm1 = np.sqrt(np.sqrt(np.sum(np.square(v1))))
    norm2 = np.sqrt(np.sqrt(np.sum(np.square(v2))))
    dot = np.dot(v1,v2)
    return dot / (norm1*norm2)

my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector,user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))

recommend_list=[]
for i , log in enumerate(zip(my_vector,best_match_vector)):
    log1,log2 = log
    if log1 <1. and log2 > 0. :
        recommend_list.append(i)
print(recommend_list)


#todo 사용자가 평가한 영화 점수 추가해서
adj_matrix = np.ndarray(shape,dtype=int)
for user_id,movie_id,rating,time in raw_data:
    adj_matrix[user_id][movie_id] = rating

my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = 9999, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        euclidean = np.sqrt(np.sum(np.square(my_vector - user_vector)))
        if euclidean < best_match:
            best_match = euclidean
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))

# 코사인
my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector,user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))

#todo 협업 필터링
# 사용자와 항목의 유사성을 동시에 고려
# 기존에 내 관심사가 아닌 항목이라도 추천 가능
# 자동으로 임베딩
# 추가 특성 사용 어려움

from surprise import KNNBasic,SVD,SVDpp,NMF
from surprise import Dataset
from surprise.model_selection import cross_validate

# knn 사용
model = KNNBasic()
cross_validate(model,data,measures=['rmse','mae'],cv=5,n_jobs=4,verbose=True)

# svd 사용
model = SVD()
cross_validate(model,data,measures=['rmse','mae'],cv=5,n_jobs=4,verbose=True)

# nmf 사용
model = NMF()
cross_validate(model,data,measures=['rmse','mae'],cv=5,n_jobs=4,verbose=True)

# svd++ 사용
model = SVDpp()
cross_validate(model,data,measures=['rmse','mae'],cv=5,n_jobs=4,verbose=True)

#TODO 하이브리드
# 컨텐츠 기반 필터링과 협업 필터링을 조합한 방식
# 많은 하이브리드 방법 존재
from sklearn.decomposition import randomized_svd,non_negative_factorization
from surprise import Dataset

raw_data = np.array(data.raw_ratings,dtype=int)
raw_data[:0] -=1
raw_data[:1] -=1

n_users = np.max(raw_data[:,0])
n_movies = np.max(raw_data[:,1])
shape = (n_users +1, n_movies+1)

adj_matrix = np.ndarray(shape,dtype=int)
for user_id,movie_id,rating,time in raw_data:
    adj_matrix[user_id][movie_id] = rating

U, S,V = randomized_svd(adj_matrix,n_components=2) # 행렬 분해 S가 특성
S = np.diag(S)

print(U.shape)
print(S.shape)
print(V.shape)

np.matmul(np.matmul(U,S),V) #행렬 다시 곱함

#TODO 사용자 기반 추천
my_id, my_vector = 0, U[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(U):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector,user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))

recommend_list=[]
for i , log in enumerate(zip(adj_matrix[my_id],adj_matrix[best_match_id])):
    log1,log2 = log
    if log1 <1. and log2 > 0. :
        recommend_list.append(i)
print(recommend_list)

#todo 항목기반 추천
# 내가 본 항목과 비슷한 항목 추천
# 항목 특징 벡터의 유사도 사용

my_id, my_vector = 0, V.T[0]
best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(U):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector,user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match: {}, Best Match ID: {}'.format(best_match, best_match_id))


