import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# Replace with your free API key from https://www.themoviedb.org/
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    movies.columns = movies.columns.str.strip().str.lower()
    return movies

movies_df = load_data()

st.title("🎬 Tamil Movie Recommender")

if 'title' not in movies_df.columns:
    st.error("Could not find 'title' column in movies.csv")
    st.stop()

# 1. Sidebar Genre Filter
st.sidebar.header("Filter Options")
if 'genres' in movies_df.columns:
    movies_df['genres'] = movies_df['genres'].fillna('')
    all_genres = sorted(list(set(g for sublist in movies_df['genres'].str.split('|') for g in sublist if g)))
    selected_genres = st.sidebar.multiselect("Filter by Genre", options=all_genres)
    
    if selected_genres:
        movies_df = movies_df[movies_df['genres'].apply(lambda x: any(g in str(x) for g in selected_genres))]

# 2. Auto-Complete Search Box
movie_titles = movies_df['title'].values if not movies_df.empty else ["No movies found"]
selected_movie = st.selectbox("Search or select a movie:", movie_titles)

# 3. Helper Function to Fetch Poster
def fetch_poster(movie_title, api_key):
    if not api_key or api_key == "YOUR_TMDB_API_KEY":
        return None
    try:
        # Remove year from title for clean TMDb search (e.g. "Kaithi (2019)" -> "Kaithi")
        clean_title = movie_title.split('(')[0].strip()
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
        response = requests.get(url, timeout=5).json()
        results = response.get('results', [])
        
        if results and results[0].get('poster_path'):
            poster_path = results[0]['poster_path']
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return None

# 4. Recommendation Function (ONLY includes movies with valid posters)
def get_recommendations_with_posters(target_movie, df, api_key, top_n=4):
    if 'genres' not in df.columns or len(df) <= 1:
        return []
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    
    idx_list = df[df['title'] == target_movie].index
    if len(idx_list) == 0:
        return []
        
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, score in sim_scores:
        candidate_title = df.iloc[i]['title']
        
        # Skip the selected movie itself
        if candidate_title == target_movie:
            continue
            
        poster_url = fetch_poster(candidate_title, api_key)
        
        # ONLY include the movie if a poster was successfully found
        if poster_url:
            recommendations.append((candidate_title, poster_url))
            
        if len(recommendations) == top_n:
            break
            
    return recommendations

# 5. Display Recommendations
if st.button("Get Recommendations"):
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        st.error("Please replace 'YOUR_TMDB_API_KEY' in app.py with your TMDb API key to enable poster filtering.")
    else:
        st.write(f"### Recommended movies based on **{selected_movie}**:")
        
        recs = get_recommendations_with_posters(selected_movie, movies_df, TMDB_API_KEY, top_n=4)
        
        if recs:
            cols = st.columns(len(recs))
            for idx, col in enumerate(cols):
                title, poster = recs[idx]
                with col:
                    st.image(poster, use_container_width=True)
                    st.caption(f"**{title}**")
        else:
            st.warning("No recommendations found with available posters.")

# 6. User Feedback
st.divider()
st.subheader("Was this recommendation helpful?")
feedback = st.feedback("thumbs")
if feedback is not None:
    st.success("Thank you for your feedback!")
