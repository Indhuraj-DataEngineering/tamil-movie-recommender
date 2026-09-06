import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    # Clean column names (strip whitespace and convert to lowercase)
    movies.columns = movies.columns.str.strip().str.lower()
    return movies

movies_df = load_data()

st.title("🎬 Tamil Movie Recommender")

# Safety check if 'title' column is missing under any variation
if 'title' not in movies_df.columns:
    st.error(f"Could not find a 'title' column in movies.csv. Available columns in your file are: {list(movies_df.columns)}")
    st.stop()

# 1. Sidebar Genre Filter
st.sidebar.header("Filter Options")
if 'genres' in movies_df.columns:
    all_genres = sorted(list(set(g for sublist in movies_df['genres'].dropna().str.split('|') for g in sublist)))
    selected_genres = st.sidebar.multiselect("Filter by Genre", options=all_genres)
    
    if selected_genres:
        movies_df = movies_df[movies_df['genres'].apply(lambda x: any(g in str(x) for g in selected_genres))]

# 2. Auto-Complete Search Box
movie_titles = movies_df['title'].values if not movies_df.empty else ["No movies found"]
selected_movie = st.selectbox("Search or select a movie:", movie_titles)

# 3. TMDb API Helper Function for Movie Posters
def fetch_poster(movie_title, api_key="YOUR_TMDB_API_KEY"):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(url).json()
        poster_path = response['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Generate Recommendations
if st.button("Get Recommendations"):
    st.write(f"### Recommended movies based on **{selected_movie}**:")
    
    recommendations = movies_df['title'].dropna().head(4).tolist()
    
    cols = st.columns(len(recommendations)) if recommendations else []
    for idx, col in enumerate(cols):
        with col:
            rec_title = recommendations[idx]
            st.caption(rec_title)

# 4. User Feedback Widget
st.divider()
st.subheader("Was this recommendation helpful?")
feedback = st.feedback("thumbs")
if feedback is not None:
    st.success("Thank you for your feedback!")
