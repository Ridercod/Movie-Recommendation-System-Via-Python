#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(1)


# In[16]:


credits.head(1)


# In[17]:


movies = movies.merge(credits, on = 'title')


# In[18]:


movies.head(1)


# In[19]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[20]:


movies.head()


# In[21]:


import ast


# In[22]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[23]:


movies.dropna(inplace=True)


# In[24]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[25]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[26]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[27]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[28]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[29]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[30]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[31]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[32]:


movies.sample(5)


# In[33]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[34]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[35]:


movies.head()


# In[36]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[37]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[38]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[39]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


vector = cv.fit_transform(new['tags']).toarray()


# In[42]:


vector.shape


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


similarity = cosine_similarity(vector)


# In[45]:


similarity


# In[46]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[47]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        


# In[ ]:


recommend('Gandhi')


# In[ ]:


import pickle


# In[49]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:




