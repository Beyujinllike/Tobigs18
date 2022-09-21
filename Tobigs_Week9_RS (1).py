#!/usr/bin/env python
# coding: utf-8

# In[471]:


##전처리 및 EDA

##데이터셋 특징
##5점 만점
##100836 ratings and 3683 tag applications across 9742 movies(610 users)
##no demographic information

#movie.csv:영화 제목,장르,movie id
#ratings.csv:평가 점수,timestamp
#tags.csv:tag,timestamp(tag는 사용자가 생성한 metadata/주관적)


# In[ ]:


#참고:
#https://github.com/lsjsj92/recommender_system_with_Python/blob/master/004.%20recommender%20system%20basic%20with%20Python%20-%203%20Matrix%20Factorization.ipynb
#https://beckernick.github.io/matrix-factorization-recommender/
#https://github.com/MoMkhani/MovieLens-Matrix-Factorization/blob/main/Notebook/MovieLens_Matrix_Factorization.ipynb
#https://simonezz.tistory.com/23
#https://www.kaggle.com/code/ecemboluk/recommendation-system-with-cf-using-knn/notebook


# In[472]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# In[505]:


movie = pd.read_csv('movies.csv')
rating = pd.read_csv('ratings.csv')

tag = pd.read_csv('tags.csv')


# In[474]:


movie.head()
#장르 중복 표기 가능
#genre1,2,3,4로 나누어 표기해야 하나?|를 기준으로 나눠서?


# In[475]:


movie['title'].nunique()


# In[476]:


rating.head()


# In[477]:


tags.head()
#긍,부정 평가에 사용?


# In[478]:


movie.info()


# In[479]:


rating.info()


# In[480]:


rating['rating'].mean()
#평균이 3.5이니 추천 시스템이므로 무난히 평균 이상의 점수를 받은 영화만을 추천 대상으로 지정.


# In[481]:


#장르별 속성 정렬
movie["genres"].value_counts()


# In[482]:


unique_genre_dict = {}
for index, row in movie.iterrows():
    
    genre_combination = row['genres']
    parsed_genre = genre_combination.split('|')
    
    for genre in parsed_genre:
        if genre in unique_genre_dict:
            unique_genre_dict[genre] += 1
        else:
            unique_genre_dict[genre] = 1   


# In[483]:


plt.rcParams['figure.figsize'] = [20,16]
sns.barplot(list(unique_genre_dict.keys()), list(unique_genre_dict.values()),
           alpha=0.8)
plt.title('Popular genre in movies')
plt.ylabel('Count of genre', fontsize=12)
plt.xlabel('Genre', fontsize=12)
plt.show()
#평균평점이상의 영화 중 drama,comedy가 주인기장르임을 확인한다.


# In[484]:


#각 영화별 평균 평점과 rating개수 확인
movie_grouped_rating_info = rating.groupby('movieId')['rating'].agg(['count','mean'])
movie_grouped_rating_info.columns = ['rated_count', 'rating_mean']


# In[485]:


movie_grouped_rating_info['rating_mean'].hist(bins=50, grid=False)


# In[486]:


movie_grouped_rating_info.head(5)


# In[512]:


#영화별 평점과 평가자 수 확인
eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[513]:


eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[514]:


eda_rating.sort_values('count of ratings', ascending=False).head(10)


# In[487]:


#Memory-based CF
#1.SVD방법

get_ipython().system('pip install scikit-surprise')
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


# In[488]:


reader = Reader(rating_scale=(1,5)) #평점 1-5
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
train_data = data.build_full_trainset()


# In[489]:


import time

train_start = time.time()
model = SVD(n_factors=8,
           lr_all=0.005,
           reg_all=0.02,
           n_epochs=100)
model.fit(train_data)
train_end = time.time()
print("training time of model: %.2f seconds" % (train_end - train_start))


# In[490]:


#딕셔너리 형태로 rating을 바꿔줌으로써 sparse피하고,계산 성능을 올려줌.
def ratings_dictionary(ratings):
    r_dict = {}
    for i in ratings[1:]:
        if i[0] not in r_dict.keys():
            r_dict[i[0]]={i[1]:float(i[2])}
        else:
            r_dict[i[0]].setdefault(i[1],float(i[2]))
    return r_dict


# In[491]:


#코사인 유사도 확인
import math
import numpy as np
def cosine_similarity(A,B): # ex) A=[2.0, 3.0] , B=[5.0, 3.5] ; 리스트 속 수치형 자료, 당연히 차원이 같아야함.
    dot_p = np.dot(A,B)
    A_norms = math.sqrt(sum([i**2 for i in A]))
    B_norms = math.sqrt(sum([i**2 for i in B]))
    AB_norms = A_norms * B_norms
    
    return dot_p / AB_norms # 1에 가까울수록 유사함.


# In[492]:


#SVD 모델 평가(RMSE 측정)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
train_data, test_data = train_test_split(data, test_size=0.2)


# In[493]:


train_start = time.time()
model = SVD(n_factors=8,
           lr_all=0.005,
           reg_all=0.02,
           n_epochs=100)
model.fit(train_data)
train_end = time.time()
print("training time of model: %.2f seconds" % (train_end - train_start))

predictions = model.test(test_data)

print("RMSE of test in SVD model:")
accuracy.rmse(predictions)
#소실값을 채우는 문제가 한계인데 해당 데이터는 소실값이 없음.


# In[494]:


#user23의 SVD추천 결과/타겟이 보지 않은 영화 중 예상 평점이 높은 10개 선정
target_user_id = 23
target_user_data = rating[rating['userId']==target_user_id]
target_user_data.head(5)


# In[495]:


target_user_movie_rating_dict = {}

for index, row in target_user_data.iterrows():
    movieId = row['movieId']
    target_user_movie_rating_dict[movieId] = row['rating']

print(target_user_movie_rating_dict)


# In[496]:


test_data = [] #test data는 user23이 아직 보지 않은 영화들의 리스트
for index, row in movie.iterrows():
    movie_id = row['movieId']
    rating = 0
    if movie_id in target_user_movie_rating_dict:
        continue
    test_data.append((target_user_id, movieId, rating))
    


# In[497]:


target_user_predictions = model.test(test_data)


# In[498]:


def get_user_predicted_ratings(predictions, user_id, user_history):
    target_user_movie_predict_dict = {}
    for uid, mid, rating, predicted_rating, _ in predictions:
        if user_id == uid:
            if mid not in user_history:
                target_user_movie_predict_dict[mid] = predicted_rating
    return target_user_movie_predict_dict

target_user_movie_predict_dict = get_user_predicted_ratings(predictions=target_user_predictions, 
                                                                        user_id=target_user_id, 
                                                                        user_history=target_user_movie_rating_dict)   


# In[499]:


import operator

target_user_top10_predicted = sorted(target_user_movie_predict_dict.items(), 
                                             key=operator.itemgetter(1), reverse=True)[:10]


# In[500]:


target_user_top10_predicted


# In[501]:


movie_dict = {}
for index, row in movie.iterrows():
    movieId = row['movieId']
    movie_title = row['title']
    movie_dict[movie_id] = movie_title
    


# In[539]:


for predicted in target_user_top10_predicted:
    movie_id = predicted[0]
    predicted_rating = predicted[1]
    print(movie_dict[movie_id], ":", predicted_rating)


# In[605]:


#Model_based2:SVD기반의 matrix factorization
#전략:factorization과 neighborhood methods를 하나의 framework로 결합
from scipy.sparse.linalg import svds

# Merge two datasets to have better picture
df = pd.merge(rating, movie, on='movieId')
df.head()


# In[606]:


mtrx_df = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
mtrx_df.head()
#pivot table생성


# In[607]:


#de-mean data
mtrx = mtrx_df.to_numpy()
ratings_mean = np.mean(mtrx, axis = 1)
normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)


# In[608]:


normalized_mtrx


# In[609]:


#Singular value decomposition
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)


# In[610]:


sigma = np.diag(sigma)


# In[537]:


#prediction
all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)


# In[ ]:


def recommend_movies(preds_df, userId, movie, ratings_df, num_recommendations=5):
    '''Recommend top K movies to any chosen user

    Args:
    preds_df: prediction dataframe obtained from matrix factorization
    userId: chosen user
    movie: movie dataframe
    ratings_df: rating dataframe
    num_recommendations: desired number of recommendations

    Return:
    user_rated: movies that user already rated
    recommendations: final recommendations

    '''
    # Get user id, keep in mind index starts from zero
    user_row_number = userId-1 
    # Sort user's predictons
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) 
    # List movies user already rated
    user_data = ratings_df[ratings_df.userId == (userId)]
    user_rated = (user_data.merge(movie, how = 'left', left_on = 'movieId', right_on = 'movieId').
                  sort_values(['rating'], ascending=False)
                 )
    
    # f'User {userId} has already rated {user_rated.shape[0]} films.'

    recommendations = (movie[~movie['movieId'].isin(user_rated['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
               rename(columns = {user_row_number: 'Predictions'}).
               sort_values('Predictions', ascending = False).
               iloc[:num_recommendations, :-1]
                      )

    return user_rated, recommendations


# In[ ]:


#이미 평가된 항목 보여주고,random choice기반하여 추천
already_rated, predictions = recommend_movies(preds_df, 100, movie, rating, 10)
# 이미 평가된 영화항목
already_rated.head(10)


# In[ ]:


# Recommendations for desired user
predictions


# In[589]:


#Memory_based:Centered KNN
from surprise import Dataset
from surprise import Reader


# In[590]:


#item-based similarity사용
from scipy.sparse import csr_matrix
# pivot ratings into movie features
user_movie_table = df.pivot_table(index = ["title"],columns = ["userId"],values = "rating").fillna(0)
# convert dataframe of movie features to scipy sparse matrix
mat_movie_features = csr_matrix(df_movie_features.values)


# In[591]:


from sklearn.neighbors import NearestNeighbors
query_index = np.random.choice(user_movie_table.shape[0])

user_movie_table_matrix = csr_matrix(user_movie_table.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(user_movie_table_matrix)
distances, indices = model_knn.kneighbors(user_movie_table.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6)


# In[592]:


movie = []
distance = []

for i in range(0, len(distances.flatten())):
    if i != 0:
        movie.append(user_movie_table.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])    

m=pd.Series(movie,name='movie')
d=pd.Series(distance,name='distance')
recommend = pd.concat([m,d], axis=1)
recommend = recommend.sort_values('distance',ascending=False)

print('Recommendations for {0}:\n'.format(user_movie_table.index[query_index]))
for i in range(0,recommend.shape[0]):
    print('{0}: {1}, with distance of {2}'.format(i, recommend["movie"].iloc[i], recommend["distance"].iloc[i]))

