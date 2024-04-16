import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import requests

movies = pickle.load(open("movielist.pkl","rb"))
vectors = pickle.load(open("vectors.pkl","rb"))


def fetch_poster(movie_id):
    data = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=a7cdaf4cd0359210b90afdd4fe28b356&language=en-US'.format(movie_id))
    data =data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path

movie_list=movies['MovieName'].values

st.header("Movie Recommender System")

import streamlit.components.v1 as components

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")


imageUrls = [
    fetch_poster(1632),
    fetch_poster(299536),
    fetch_poster(17455),
    fetch_poster(2830),
    fetch_poster(429422),
    fetch_poster(9722),
    fetch_poster(13972),
    fetch_poster(240),
    fetch_poster(155),
    fetch_poster(598),
    fetch_poster(914),
    fetch_poster(255709),
    fetch_poster(572154)
   
    ]


imageCarouselComponent(imageUrls=imageUrls, height=200)

selectvalue = st.selectbox(
    'Select Movie from dropdown',
    movie_list
)

k = 6
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(vectors)






def recommend(movie):
    movie_index = movies[movies['MovieName'] == movie].index[0]
    target_movie = vectors[movie_index]

    _, top_movie_indices = knn_model.kneighbors(target_movie)
    distances,_= knn_model.kneighbors(target_movie)

    ## Flatten the distance array andtarget_movie = vectors[movie_index] top_movie_indices array
    distances = distances.flatten()
    top_movie_indices = top_movie_indices.flatten()
    top_similar_movies = movies.iloc[top_movie_indices]['MovieName']

    recommend_movies = []
    recommend_poster = []
    for i in top_similar_movies:
        movie_id = movies[movies['MovieName'] == i]['id'].values[0]
        recommend_movies.append(i)
        recommend_poster.append(fetch_poster(movie_id))
    return recommend_movies,recommend_poster



if st.button("Show Recommend"):
    movie_name,movie_poster = recommend(selectvalue)
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1:
        st.text(movie_name[0])
        st.image(movie_poster[0])
    with col2:
        st.text(movie_name[1])
        st.image(movie_poster[1])
    with col3:
        st.text(movie_name[2])
        st.image(movie_poster[2])
    with col4:
        st.text(movie_name[3])
        st.image(movie_poster[3])
    with col5:
        st.text(movie_name[4])          
        st.image(movie_poster[4])
    with col6:
        st.text(movie_name[5])          
        st.image(movie_poster[5])                
