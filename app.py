import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page layout
st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# Load and clean dataset
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

# 3. Recommendation Function (Excludes the selected movie)
def get_recommendations(target_movie, df, top_n=4):
    # Exclude the selected movie so it doesn't recommend itself
    filtered_df = df[df['title'] != target_movie].copy()
    
    if 'genres' in df.columns and len(df) > 1:
        # Calculate genre similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['genres'])
        
        idx_list = df[df['title'] == target_movie].index
        if len(idx_list) > 0:
            idx = idx_list[0]
            sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Remove the selected movie index from recommendations
            sim_scores = [x for x in sim_scores if x[0] != idx]
            
            movie_indices = [i[0] for i in sim_scores[:top_n]]
            return df.iloc[movie_indices]['title'].tolist()
            
    return filtered_df['title'].head(top_n).tolist()

# 4. Display Recommendations
if st.button("Get Recommendations"):
    st.write(f"### Recommended movies based on **{selected_movie}**:")
    
    recs = get_recommendations(selected_movie, movies_df, top_n=4)
    
    if recs:
        cols = st.columns(len(recs))
        for idx, col in enumerate(cols):
            with col:
                st.write(f"**{recs[idx]}**")
    else:
        st.write("No other movies found.")

# 5. User Feedback
st.divider()
st.subheader("Was this recommendation helpful?")
feedback = st.feedback("thumbs")
if feedback is not None:
    st.success("Thank you for your feedback!")
