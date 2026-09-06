import streamlit as st
import pandas as pd
import requests

# Set page layout
st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# Cache data loading for faster performance
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    return movies

movies_df = load_data()

st.title("🎬 Tamil Movie Recommender")

# 1. Sidebar Genre & Release Filter
st.sidebar.header("Filter Options")
if 'genres' in movies_df.columns:
    all_genres = sorted(list(set(g for sublist in movies_df['genres'].dropna().str.split('|') for g in sublist)))
    selected_genres = st.sidebar.multiselect("Filter by Genre", options=all_genres)
    
    # Filter movie list based on genre selection
    if selected_genres:
        movies_df = movies_df[movies_df['genres'].apply(lambda x: any(g in str(x) for g in selected_genres))]

# 2. Auto-Complete Search Box
movie_titles = movies_df['title'].values if not movies_df.empty else ["No movies found"]
selected_movie = st.selectbox("Search or select a movie:", movie_titles)

# 3. TMDb API Helper Function for Movie Posters
def fetch_poster(movie_title, api_key="YOUR_TMDB_API_KEY"):
    """Fetches movie poster URL from TMDb API."""
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
    
    # Placeholder recommendation list (replace with your similarity model output)
    recommendations = movies_df['title'].dropna().head(4).tolist()
    
    # Display in columns with poster images
    cols = st.columns(len(recommendations))
    for idx, col in enumerate(cols):
        with col:
            rec_title = recommendations[idx]
            st.caption(rec_title)
            # Uncomment line below once you insert your TMDb API key:
            # st.image(fetch_poster(rec_title))

# 4. User Feedback Widget
st.divider()
st.subheader("Was this recommendation helpful?")
feedback = st.feedback("thumbs")
if feedback is not None:
    st.success("Thank you for your feedback!")
