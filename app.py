import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# Reads API Key from Streamlit Cloud Secrets
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

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

# Sidebar Genre Filter
st.sidebar.header("Filter Options")
if 'genres' in movies_df.columns:
    movies_df['genres'] = movies_df['genres'].fillna('')
    all_genres = sorted(list(set(g for sublist in movies_df['genres'].str.split('|') for g in sublist if g)))
    selected_genres = st.sidebar.multiselect("Filter by Genre", options=all_genres)
    
    if selected_genres:
        movies_df = movies_df[movies_df['genres'].apply(lambda x: any(g in str(x) for g in selected_genres))]

# Search Box
movie_titles = movies_df['title'].values if not movies_df.empty else ["No movies found"]
selected_movie = st.selectbox("Search or select a movie:", movie_titles)

# Helper Function: Fetches Poster, TMDb Rating, and Release Year
@st.cache_data(ttl=86400)
def fetch_movie_details(movie_title, api_key):
    if not api_key:
        return None
    try:
        clean_title = movie_title.split('(')[0].strip()
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}&language=en-US&region=IN"
        response = requests.get(url, timeout=5).json()
        results = response.get('results', [])
        
        if results and results[0].get('poster_path'):
            first_match = results[0]
            return {
                "poster": f"https://image.tmdb.org/t/p/w500/{first_match['poster_path']}",
                "rating": round(first_match.get('vote_average', 0.0), 1),
                "year": first_match.get('release_date', '')[:4]
            }
    except Exception:
        pass
    return None

# Recommendation Logic (Includes only movies with valid TMDb details)
def get_recommendations_with_details(target_movie, df, api_key, top_n=4):
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
        
        if candidate_title == target_movie:
            continue
            
        details = fetch_movie_details(candidate_title, api_key)
        
        if details:
            recommendations.append((candidate_title, details))
            
        if len(recommendations) == top_n:
            break
            
    return recommendations

# Render UI
if st.button("Get Recommendations"):
    if not TMDB_API_KEY:
        st.error("TMDb API key missing in Streamlit secrets.")
    else:
        st.write(f"### Recommended movies based on **{selected_movie}**:")
        
        recs = get_recommendations_with_details(selected_movie, movies_df, TMDB_API_KEY, top_n=4)
        
        if recs:
            cols = st.columns(len(recs))
            for idx, col in enumerate(cols):
                title, details = recs[idx]
                with col:
                    # Bordered Card Container Layout
                    with st.container(border=True):
                        st.image(details['poster'], use_container_width=True)
                        st.markdown(f"**{title}**")
                        
                        # Rating & Release Year Badges
                        col_rating, col_year = st.columns(2)
                        with col_rating:
                            st.caption(f"⭐ **{details['rating']}/10**")
                        with col_year:
                            st.caption(f"📅 **{details['year'] if details['year'] else 'N/A'}**")
        else:
            st.warning("No recommendations found with available posters.")

# Feedback Widget
st.divider()
st.subheader("Was this recommendation helpful?")
feedback = st.feedback("thumbs")
if feedback is not None:
    st.success("Thank you for your feedback!")
