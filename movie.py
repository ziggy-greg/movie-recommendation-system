import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
ratings = pd.read_csv("u.data", sep='\t', names=["user_id", "movie_id", "rating", "timestamp"])
movies = pd.read_csv("u.item", sep='|', encoding="ISO-8859-1", usecols=[0, 1], names=["movie_id", "title"], engine='python')
df = pd.merge(ratings, movies, on="movie_id")

# Create pivot table for users and movies
pivot_table = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(pivot_table)
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

# TMDB API Setup
api_key = "87ca1a58d8af64e3bea559458135cde2"
def fetch_movie_details(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    response = requests.get(url)
    data = response.json()

    if 'results' in data and data['results']:  # Safe check
        movie = data['results'][0]
        poster_path = movie.get('poster_path')
        overview = movie.get('overview')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return poster_url, overview
    return None, "No overview found."

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("\U0001F3AC Movie Recommendation System")

user_id_input = st.number_input("Select User ID to Recommend For", min_value=1, max_value=int(df.user_id.max()), step=1)

if st.button("Get Recommendations"):
    if user_id_input not in user_similarity_df.index:
        st.warning("User ID not found.")
    else:
        similar_users = user_similarity_df[user_id_input].sort_values(ascending=False).drop(user_id_input)
        top_similar_user_id = similar_users.index[0]

        user_movies = df[df['user_id'] == user_id_input]['movie_id'].unique()
        similar_user_movies = df[df['user_id'] == top_similar_user_id]
        unseen_recommendations = similar_user_movies[~similar_user_movies['movie_id'].isin(user_movies)]

        top_recs = unseen_recommendations.groupby('movie_id').agg({'rating': 'mean'}).sort_values(by='rating', ascending=False).head(10)
        top_movie_ids = top_recs.index.tolist()

        movie_id_to_title = dict(zip(movies.movie_id, movies.title))

        for movie_id in top_movie_ids:
            title = movie_id_to_title.get(movie_id, "Unknown Title")
            avg_rating = round(top_recs.loc[movie_id, 'rating'], 1)
            poster_url, overview = fetch_movie_details(title)

            st.subheader(f"{title} ({avg_rating}⭐)")
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




