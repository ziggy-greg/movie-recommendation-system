import streamlit as st
import pandas as pd
import requests
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    ratings = pd.read_csv("u.data", sep='\t', names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv("u.item", sep='|', encoding="ISO-8859-1", usecols=[0, 1], names=["movie_id", "title"], engine='python')
    return pd.merge(ratings, movies, on="movie_id"), movies

df, movies = load_data()

# Build surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train model
model = SVD()
model.fit(trainset)

# Build movie-id to title mapping
movie_id_to_title = dict(zip(movies.movie_id, movies.title))

# TMDB API Setup
api_key = os.getenv("TMDB_API_KEY", "87ca1a58d8a6f4e3bea559458135cde2")
def fetch_movie_details(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        movie = data['results'][0]
        poster_path = movie.get('poster_path')
        overview = movie.get('overview')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return poster_url, overview
    return None, "No overview found."

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("\U0001F3AC Movie Recommendation System")

user_id_input = st.number_input("Select User ID to Recommend For", min_value=1, max_value=1000, step=1)

if st.button("Get Recommendations"):
    user_movies = df[df['user_id'] == user_id_input]['movie_id'].unique()
    all_movie_ids = df['movie_id'].unique()
    unseen_movies = [movie for movie in all_movie_ids if movie not in user_movies]

    predictions = [model.predict(user_id_input, movie_id) for movie_id in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:10]

    for pred in top_predictions:
        movie_id = pred.iid
        title = movie_id_to_title.get(movie_id, "Unknown Title")
        poster_url, overview = fetch_movie_details(title)

        st.subheader(f"{title} ({round(pred.est, 1)}⭐)")
        col1, col2 = st.columns([1, 3])
        with col1:
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.write("No poster available")
        with col2:
            st.write(overview)

st.markdown("---")
st.markdown("Made by Ziggy Greg | [GitHub](https://github.com/ziggy-greg)")



