import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    background-color: #07070f !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #07070f !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem 4rem !important; max-width: 720px !important; }

/* ── Ambient blobs ── */
.stApp::before {
    content: '';
    position: fixed; top: -150px; right: -150px;
    width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(224,123,84,.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

/* ── Header ── */
.header-badge {
    display: inline-block;
    background: rgba(224,123,84,.12);
    border: 1px solid rgba(224,123,84,.3);
    color: #e07b54;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
    font-family: 'Courier New', monospace;
}
.main-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(32px, 6vw, 48px) !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
}
.gradient-word {
    background: linear-gradient(90deg, #e07b54, #f5c518);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #555 !important;
    font-size: 14px !important;
    margin-bottom: 32px !important;
}

/* ── Select box ── */
.stSelectbox label {
    color: #666 !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}
.stSelectbox > div > div {
    background: #0d0d1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    padding: 4px 4px !important;
    transition: border-color .2s !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(224,123,84,.5) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #e07b54 !important;
    box-shadow: 0 0 0 3px rgba(224,123,84,.1) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #e07b54, #c0392b) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    margin-top: 8px !important;
    box-shadow: 0 4px 20px rgba(224,123,84,.3) !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(224,123,84,.45) !important;
    background: linear-gradient(135deg, #e8855e, #cc2f2f) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Divider ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 18px 0;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    color: #fff;
    white-space: nowrap;
}
.section-line {
    height: 1px;
    flex: 1;
    background: linear-gradient(90deg, rgba(224,123,84,.4), transparent);
}
.section-count {
    color: #e07b54;
    font-size: 11px;
    font-family: 'Courier New', monospace;
    white-space: nowrap;
}

/* ── Movie card ── */
.movie-card {
    background: linear-gradient(135deg, #12121e, #0d0d1a);
    border: 1px solid #1a1a2a;
    border-left: 3px solid #e07b54;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all .25s ease;
    cursor: default;
    animation: slideIn .4s ease forwards;
    opacity: 0;
}
.movie-card:hover {
    border-color: rgba(224,123,84,.5) !important;
    border-left-color: #e07b54 !important;
    background: linear-gradient(135deg, #1a1a2e, #141428) !important;
    transform: translateX(5px);
    box-shadow: 0 6px 24px rgba(224,123,84,.12);
}

/* Delay each card */
.movie-card:nth-child(1)  { animation-delay: .05s; }
.movie-card:nth-child(2)  { animation-delay: .10s; }
.movie-card:nth-child(3)  { animation-delay: .15s; }
.movie-card:nth-child(4)  { animation-delay: .20s; }
.movie-card:nth-child(5)  { animation-delay: .25s; }
.movie-card:nth-child(6)  { animation-delay: .30s; }
.movie-card:nth-child(7)  { animation-delay: .35s; }
.movie-card:nth-child(8)  { animation-delay: .40s; }
.movie-card:nth-child(9)  { animation-delay: .45s; }
.movie-card:nth-child(10) { animation-delay: .50s; }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-18px); }
    to   { opacity: 1; transform: translateX(0); }
}

.card-left { flex: 1; }
.card-rank {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #e07b54;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 6px;
}
.card-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.tag {
    background: rgba(224,123,84,.12);
    border: 1px solid rgba(224,123,84,.25);
    color: #e07b54;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'Courier New', monospace;
    letter-spacing: .5px;
    text-transform: uppercase;
}
.similarity-badge {
    background: rgba(245,197,24,.08);
    border: 1px solid rgba(245,197,24,.2);
    color: #f5c518;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'Courier New', monospace;
}

.card-score {
    margin-left: 18px;
    text-align: center;
    min-width: 52px;
}
.score-circle {
    width: 50px; height: 50px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    position: relative;
    font-size: 11px; font-weight: 700;
    color: #e07b54;
    font-family: 'Courier New', monospace;
}
.score-label {
    font-size: 9px;
    color: #444;
    letter-spacing: 1px;
    margin-top: 3px;
    font-family: 'Courier New', monospace;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    color: #2a2a3a;
    font-size: 10px;
    letter-spacing: 2px;
    font-family: 'Courier New', monospace;
    margin-top: 32px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("movie_dataset.csv")
    df['overview'] = df['overview'].fillna('')

    # Combine overview + genres for richer similarity
    features = df['overview']
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('')
        features = df['overview'] + ' ' + df['genres']

    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(features)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

df, cosine_sim = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-badge">🎬 &nbsp;AI-POWERED</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="main-title">Movie<br>'
    '<span class="gradient-word">Recommendation</span><br>System</h1>',
    unsafe_allow_html=True
)
st.markdown('<p class="subtitle">Apni pasandida film chunein — baaki hum karein</p>', unsafe_allow_html=True)


# ── Selector ──────────────────────────────────────────────────────────────────
movie_name = st.selectbox("FILM CHUNEIN", df['title'].values)

recommend_clicked = st.button("🎯  RECOMMEND KARO")


# ── Recommendations ───────────────────────────────────────────────────────────
if recommend_clicked:
    idx = df[df['title'] == movie_name].index[0]
    distances = cosine_sim[idx]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    # Section header
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">Recommended Films</span>
        <div class="section-line"></div>
        <span class="section-count">{len(movies_list)} results</span>
    </div>
    """, unsafe_allow_html=True)

    # Cards
    for rank, (movie_idx, score) in enumerate(movies_list, 1):
        row = df.iloc[movie_idx]
        title = row['title']
        pct = round(score * 100)

        # Optional columns
        genre = ""
        if 'genres' in df.columns and row.get('genres', ''):
            raw = str(row['genres'])
            genre = raw.split()[0] if raw else ""

        year = ""
        for col in ['release_date', 'year', 'release_year']:
            if col in df.columns:
                val = str(row.get(col, ''))
                if len(val) >= 4:
                    year = val[:4]
                    break

        tag_html = f'<span class="tag">{genre}</span>' if genre else ""
        score_style = (
            f"background: conic-gradient(#e07b54 {pct*3.6}deg, #1a1a2e 0deg);"
        )

        st.markdown(f"""
        <div class="movie-card">
            <div class="card-left">
                <div class="card-rank">#{rank:02d} &nbsp;·&nbsp; {year}</div>
                <div class="card-title">{title}</div>
                <div class="card-meta">
                    {tag_html}
                    <span class="similarity-badge">⚡ {pct}% match</span>
                </div>
            </div>
            <div class="card-score">
                <div class="score-circle" style="{score_style}">
                    <div style="width:38px;height:38px;border-radius:50%;
                                background:#0d0d1a;display:flex;
                                align-items:center;justify-content:center;">
                        {pct}%
                    </div>
                </div>
                <div class="score-label">MATCH</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<p class="footer-text">Content-Based Filtering &nbsp;·&nbsp; TF-IDF + Cosine Similarity</p>',
        unsafe_allow_html=True
    )

