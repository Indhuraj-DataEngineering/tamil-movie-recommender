import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# --- 1. Recommended Solid Dark Background CSS ---
st.markdown(
    """
    <style>
    /* Main page solid dark background */
    .stApp {
        background-color: #0b0f17;
    }

    /* Sidebar solid background */
    [data-testid="stSidebar"] {
        background-color: #121824;
        border-right: 1px solid #1e293b;
    }

    /* Header styling */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Tamil Movie Recommender")


# --- 2. Solid Color Poster Card Generator ---
def render_solid_poster(title, genre, rating):
    # Palette of rich solid gradients for poster cards
    gradients = [
        "linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%)",  # Deep Navy
        "linear-gradient(135deg, #581c87 0%, #1e1b4b 100%)",  # Royal Violet
        "linear-gradient(135deg, #831843 0%, #31121f 100%)",  # Deep Crimson
        "linear-gradient(135deg, #064e3b 0%, #022c22 100%)",  # Dark Emerald
        "linear-gradient(135deg, #7c2d12 0%, #361104 100%)",  # Burnt Amber
        "linear-gradient(135deg, #334155 0%, #0f172a 100%)",  # Slate Graphite
    ]

    # Select a deterministic color based on the title length
    bg_style = gradients[len(title) % len(gradients)]
    formatted_genre = (
        str(genre).replace("|", " • ") if pd.notna(genre) else "Tamil Cinema"
    )

    card_html = f"""
    <div style="
        background: {bg_style};
        height: 260px;
        border-radius: 14px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 24px;">🎬</span>
            <span style="
                background: rgba(255, 255, 255, 0.15);
                color: #fbbf24;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
            ">⭐ {rating}</span>
        </div>
        <div>
            <div style="
                color: #ffffff;
                font-size: 18px;
                font-weight: 700;
                line-height: 1.3;
                margin-bottom: 6px;
            ">{title}</div>
            <div style="
                color: #94a3b8;
                font-size: 12px;
                font-weight: 500;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            ">{formatted_genre}</div>
        </div>
    </div>
    """
    return card_html


# --- 3. Data Processing & Recommendation Engine ---
@st.cache_data
def load_and_process_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies.columns = movies.columns.str.strip().str.lower().str.replace("_", "")
    ratings.columns = (
        ratings.columns.str.strip().str.lower().str.replace("_", "")
    )

    movie_renames = {c: "movieId" for c in movies.columns if "movie" in c}
    movie_renames.update({c: "title" for c in movies.columns if "title" in c})
    movie_renames.update({c: "genres" for c in movies.columns if "genre" in c})
    movies = movies.rename(columns=movie_renames)

    rating_renames = {c: "movieId" for c in ratings.columns if "movie" in c}
    rating_renames.update({c: "userId" for c in ratings.columns if "user" in c})
    rating_renames.update(
        {c: "rating" for c in ratings.columns if "rate" in c}
    )
    ratings = ratings.rename(columns=rating_renames)

    movies = movies.loc[:, ~movies.columns.duplicated()]
    ratings = ratings.loc[:, ~ratings.columns.duplicated()]

    movies = movies.dropna(subset=["title"])
    movies["title"] = movies["title"].astype(str).str.strip()
    movies = movies[
        (movies["title"] != "") & (movies["title"].str.lower() != "nan")
    ]
    movies = movies.reset_index(drop=True)

    if "rating" in ratings.columns and "movieId" in ratings.columns:
        stats = (
            ratings.groupby("movieId")
            .agg(avg_rating=("rating", "mean"), vote_count=("rating", "count"))
            .reset_index()
        )
        movies = movies.merge(stats, on="movieId", how="left")
        movies["avg_rating"] = movies["avg_rating"].fillna(0).round(1)
        movies["vote_count"] = movies["vote_count"].fillna(0).astype(int)
    else:
        movies["avg_rating"] = 4.0
        movies["vote_count"] = 1

    genres_text = (
        movies["genres"].fillna("")
        if "genres" in movies.columns
        else movies["title"]
    )
    movies["genres_clean"] = genres_text.str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    all_genres = set()
    for genre_str in movies["genres"].dropna():
        for g in str(genre_str).split("|"):
            if g.strip():
                all_genres.add(g.strip())

    return movies, cosine_sim, sorted(list(all_genres))


movies, cosine_sim, available_genres = load_and_process_data()

# --- 4. Sidebar Controls ---
st.sidebar.header("⚙️ Filter Settings")
top_n = st.sidebar.slider(
    "Number of Recommendations", min_value=1, max_value=10, value=5
)
min_rating = st.sidebar.slider(
    "Minimum User Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
)
selected_genres = st.sidebar.multiselect(
    "Filter by Genres:", options=available_genres, default=[]
)

# --- 5. Main Recommendation View ---
selected_movie = st.selectbox(
    "Select a Tamil movie you like:", movies["title"].values
)


def recommend(title, count=10):
    matches = movies[movies["title"] == str(title)].index
    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    candidate_indices = [i[0] for i in scores[1:]]
    recommendations = movies.iloc[candidate_indices].copy()

    recommendations = recommendations[
        recommendations["avg_rating"] >= min_rating
    ]

    if selected_genres:
        pattern = "|".join(selected_genres)
        recommendations = recommendations[
            recommendations["genres"]
            .astype(str)
            .str.contains(pattern, case=False, na=False)
        ]

    return recommendations.head(count)


if st.button("Get Recommendations"):
    results = recommend(selected_movie, count=top_n)

    if results.empty:
        st.warning("No movies matched your filter criteria.")
    else:
        st.subheader(f"Top {len(results)} Movies Similar to '{selected_movie}':")

        cols = st.columns(min(top_n, len(results)))
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i]:
                # Render Solid Color Poster Card
                poster_card = render_solid_poster(
                    row["title"], row.get("genres", ""), row["avg_rating"]
                )
                st.markdown(poster_card, unsafe_allow_html=True)
                st.caption(f"**Votes:** {row['vote_count']}")