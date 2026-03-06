# House-Price-Prediction
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("🎬 Movie Recommendation System")

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

data = pd.merge(movies, ratings, on="movieId")
data = data.drop_duplicates(subset="title")

# Feature creation
data["tags"] = data["genres"]

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(data["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = data[data["title"] == movie].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(data.iloc[i[0]].title)
        
    return recommended_movies


# Dropdown menu
movie_list = data["title"].values
selected_movie = st.selectbox("Select a movie", movie_list)

# Button
if st.button("Recommend"):
    
    recommendations = recommend(selected_movie)
    
    st.subheader("Recommended Movies")
    
    for movie in recommendations:
        st.write(movie)
