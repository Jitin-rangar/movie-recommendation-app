import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load movies dataset
@st.cache_data
def load_movies():
    return pd.read_csv('movies.csv')

movies = load_movies()

# Preprocess movie data
@st.cache_data
def preprocess_movies(movies):
    movies['genres'] = movies['genres'].fillna('')
    movies['combined_features'] = movies['genres']
    return movies

movies = preprocess_movies(movies)

# Generate similarity matrix
@st.cache_data
def calculate_similarity(movies):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

similarity_matrix = calculate_similarity(movies)

# Recommend movies based on content
def recommend_movies(movie_title, movies, similarity_matrix, top_n=5):
    if movie_title not in movies['title'].values:
        st.warning("Movie title not found in the dataset. Please try a different title.")
        return []

    movie_idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sorted_scores[1:top_n + 1]:
        recommendations.append(movies.iloc[idx]['title'])
    return recommendations

# Streamlit UI
st.title('Content-Based Movie Recommendation System')

movie_title = st.text_input('Enter a Movie Title:')
if st.button('Get Recommendations'):
    recommendations = recommend_movies(movie_title, movies, similarity_matrix)
    if recommendations:
        st.write('Top 5 Movie Recommendations:')
        for idx, movie in enumerate(recommendations):
            st.write(f"{idx + 1}. {movie}")
    else:
        st.write("No recommendations available.")