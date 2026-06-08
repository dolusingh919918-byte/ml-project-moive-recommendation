import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"   # public demo key – replace with your own
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
AVATAR_BASE  = "https://image.tmdb.org/t/p/w185"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #080810 !important;
    color: #e0e0e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stApp { background-color: #080810 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem !important; max-width: 1200px !important; }

/* Ambient blobs */
.stApp::before {
    content:'';
    position:fixed; top:-200px; right:-200px;
    width:600px; height:600px; border-radius:50%;
    background:radial-gradient(circle,rgba(224,123,84,.07) 0%,transparent 70%);
    pointer-events:none; z-index:0;
}
.stApp::after {
    content:'';
    position:fixed; bottom:-200px; left:-150px;
    width:500px; height:500px; border-radius:50%;
    background:radial-gradient(circle,rgba(142,68,173,.06) 0%,transparent 70%);
    pointer-events:none; z-index:0;
}

/* ── Header ── */
.badge {
    display:inline-block;
    background:rgba(224,123,84,.1);
    border:1px solid rgba(224,123,84,.3);
    color:#e07b54; font-size:10px; letter-spacing:3px;
    text-transform:uppercase; padding:4px 14px;
    border-radius:20px; margin-bottom:12px;
    font-family:'Courier New',monospace;
}
.main-title {
    font-family:'Playfair Display',serif !important;
    font-size:clamp(28px,5vw,46px) !important;
    font-weight:700 !important; line-height:1.15 !important;
    color:#fff !important; margin:0 0 6px 0 !important;
}
.grad { background:linear-gradient(90deg,#e07b54,#f5c518);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.subtitle { color:#555 !important; font-size:14px !important; margin-bottom:28px !important; }

/* ── Selectbox ── */
.stSelectbox label {
    color:#666 !important; font-size:10px !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    font-family:'Courier New',monospace !important;
}
.stSelectbox > div > div {
    background:#0d0d1a !important; border:1px solid #1e1e2e !important;
    border-radius:10px !important; color:#e0e0e0 !important;
}
.stSelectbox > div > div:focus-within {
    border-color:#e07b54 !important;
    box-shadow:0 0 0 3px rgba(224,123,84,.1) !important;
}

/* ── Button ── */
.stButton > button {
    width:100% !important;
    background:linear-gradient(135deg,#e07b54,#c0392b) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; padding:14px 0 !important;
    font-size:15px !important; font-weight:600 !important;
    letter-spacing:1px !important;
    box-shadow:0 4px 20px rgba(224,123,84,.3) !important;
    transition:all .2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(224,123,84,.45) !important;
}

/* ── Selected movie hero card ── */
.hero-card {
    display:flex; gap:22px; align-items:flex-start;
    background:linear-gradient(135deg,#12121e,#0d0d1a);
    border:1px solid #1e1e2e; border-left:4px solid #e07b54;
    border-radius:14px; padding:20px; margin-bottom:30px;
}
.hero-card img { border-radius:10px; width:120px; min-width:120px; object-fit:cover; }
.hero-info h2 {
    font-family:'Playfair Display',serif;
    font-size:22px; color:#fff; margin:0 0 6px 0;
}
.hero-info p { color:#888; font-size:13px; line-height:1.6; margin:0 0 10px 0; }
.hero-meta { display:flex; gap:8px; flex-wrap:wrap; }
.pill {
    background:rgba(224,123,84,.1); border:1px solid rgba(224,123,84,.25);
    color:#e07b54; font-size:10px; padding:3px 10px;
    border-radius:20px; font-family:'Courier New',monospace;
    text-transform:uppercase; letter-spacing:.5px;
}
.pill-yellow {
    background:rgba(245,197,24,.08); border:1px solid rgba(245,197,24,.2);
    color:#f5c518;
}

/* ── Section header ── */
.sec-head {
    display:flex; align-items:center; gap:12px; margin:0 0 18px 0;
}
.sec-title { font-family:'Playfair Display',serif; font-size:20px; color:#fff; white-space:nowrap; }
.sec-line { height:1px; flex:1; background:linear-gradient(90deg,rgba(224,123,84,.4),transparent); }
.sec-count { color:#e07b54; font-size:11px; font-family:'Courier New',monospace; white-space:nowrap; }

/* ── Rec card ── */
.rec-card {
    background:linear-gradient(135deg,#12121e,#0d0d1a);
    border:1px solid #1a1a2a; border-radius:14px;
    overflow:hidden; transition:all .25s ease;
    animation: fadeUp .45s ease forwards; opacity:0;
}
.rec-card:hover {
    border-color:rgba(224,123,84,.45) !important;
    transform:translateY(-4px);
    box-shadow:0 12px 36px rgba(224,123,84,.12);
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.rec-poster { width:100%; aspect-ratio:2/3; object-fit:cover; display:block; }
.rec-poster-ph {
    width:100%; aspect-ratio:2/3;
    background:linear-gradient(135deg,#1a1a2e,#0d0d1a);
    display:flex; align-items:center; justify-content:center;
    font-size:36px; color:#2a2a3a;
}
.rec-body { padding:14px; }
.rec-rank {
    font-family:'Courier New',monospace; font-size:10px;
    color:#e07b54; letter-spacing:1px; margin-bottom:4px;
}
.rec-title {
    font-family:'Playfair Display',serif; font-size:15px;
    font-weight:600; color:#f0f0f0; margin-bottom:8px;
    line-height:1.3;
}
.rec-meta { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
.rec-cast { font-size:11px; color:#666; line-height:1.5; }
.rec-cast span { color:#999; }

/* ── Cast row ── */
.cast-row { display:flex; gap:8px; margin-top:10px; }
.cast-chip {
    display:flex; flex-direction:column; align-items:center; gap:4px;
    min-width:52px; max-width:60px;
}
.cast-chip img {
    width:44px; height:44px; border-radius:50%; object-fit:cover;
    border:2px solid #2a2a3a;
}
.cast-chip .cast-ph {
    width:44px; height:44px; border-radius:50%;
    background:#1a1a2e; display:flex; align-items:center;
    justify-content:center; font-size:16px; border:2px solid #2a2a3a;
}
.cast-chip span {
    font-size:9px; color:#666; text-align:center; line-height:1.2;
    word-break:break-word;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color:#e07b54 !important; }

/* ── Footer ── */
.foot {
    text-align:center; color:#222; font-size:10px;
    letter-spacing:2px; font-family:'Courier New',monospace;
    margin-top:40px; text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TMDB HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def tmdb_search(title: str):
    try:
        r = requests.get(
            f"{TMDB_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
            timeout=5
        )
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_credits(tmdb_id: int):
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}/credits",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        cast = r.json().get("cast", [])[:5]
        return cast
    except Exception:
        return []

def poster_url(path):
    return f"{POSTER_BASE}{path}" if path else None

def avatar_url(path):
    return f"{AVATAR_BASE}{path}" if path else None


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("movie_dataset.csv")
    df['overview'] = df['overview'].fillna('')
    features = df['overview']
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('')
        features = df['overview'] + ' ' + df['genres']
    tfidf   = TfidfVectorizer(stop_words='english', max_features=10000)
    mat     = tfidf.fit_transform(features)
    cos_sim = cosine_similarity(mat, mat)
    return df, cos_sim

df, cosine_sim = load_data()


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="badge">🎬 &nbsp;AI-POWERED</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="main-title">Movie <span class="grad">Recommendation</span> System</h1>',
    unsafe_allow_html=True
)
st.markdown('<p class="subtitle">Select a movie and discover films you\'ll love — powered by ML</p>',
            unsafe_allow_html=True)

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    movie_name = st.selectbox("SELECT A MOVIE", df['title'].values, label_visibility="visible")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_clicked = st.button("🎯  Find Movies")


# ─────────────────────────────────────────────
#  SELECTED MOVIE HERO
# ─────────────────────────────────────────────
if movie_name:
    with st.spinner("Loading movie info..."):
        info = tmdb_search(movie_name)

    if info:
        tmdb_id   = info.get("id")
        poster    = poster_url(info.get("poster_path"))
        overview  = info.get("overview", df[df['title']==movie_name]['overview'].values[0])
        rating    = info.get("vote_average", 0)
        release   = info.get("release_date", "")[:4]
        genres_raw = df[df['title']==movie_name]['genres'].values[0] if 'genres' in df.columns else ""

        poster_html = f'<img src="{poster}" alt="poster">' if poster else '<div style="width:120px;height:180px;background:#1a1a2e;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:32px;">🎬</div>'

        st.markdown(f"""
        <div class="hero-card">
            {poster_html}
            <div class="hero-info">
                <h2>{movie_name}</h2>
                <p>{overview[:200]}{'...' if len(overview)>200 else ''}</p>
                <div class="hero-meta">
                    <span class="pill">{release}</span>
                    <span class="pill pill-yellow">⭐ {rating:.1f}/10</span>
                    {''.join(f'<span class="pill">{g.strip()}</span>' for g in genres_raw.split() if g.strip())[:3]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RECOMMENDATIONS
# ─────────────────────────────────────────────
if recommend_clicked:
    idx = df[df['title'] == movie_name].index[0]
    movies_list = sorted(
        list(enumerate(cosine_sim[idx])),
        reverse=True, key=lambda x: x[1]
    )[1:11]

    st.markdown(f"""
    <div class="sec-head">
        <span class="sec-title">Recommended Films</span>
        <div class="sec-line"></div>
        <span class="sec-count">{len(movies_list)} results</span>
    </div>
    """, unsafe_allow_html=True)

    # 5-column grid
    cols = st.columns(5)

    for rank, (midx, score) in enumerate(movies_list):
        row   = df.iloc[midx]
        title = row['title']
        pct   = round(score * 100)
        delay = rank * 0.06

        with st.spinner(f"Fetching {title}...") if rank == 0 else st.empty():
            tmdb   = tmdb_search(title)
            cast   = tmdb_credits(tmdb["id"]) if tmdb else []

        poster_path = tmdb.get("poster_path") if tmdb else None
        release_yr  = (tmdb.get("release_date","")[:4]) if tmdb else ""
        vote        = tmdb.get("vote_average", 0) if tmdb else 0

        poster_html = (
            f'<img class="rec-poster" src="{poster_url(poster_path)}" alt="{title}">'
            if poster_path else
            '<div class="rec-poster-ph">🎬</div>'
        )

        # Cast chips
        cast_chips = ""
        for actor in cast[:4]:
            ap = avatar_url(actor.get("profile_path"))
            name = actor.get("name","")
            first = name.split()[0] if name else ""
            img_html = (
                f'<img src="{ap}" alt="{first}">'
                if ap else
                '<div class="cast-ph">👤</div>'
            )
            cast_chips += f'<div class="cast-chip">{img_html}<span>{first}</span></div>'

        genre_tags = ""
        if 'genres' in df.columns:
            graw = str(row.get('genres',''))
            genre_tags = ''.join(
                f'<span class="pill" style="font-size:9px;padding:2px 7px;">{g.strip()}</span>'
                for g in graw.split()[:2] if g.strip()
            )

        card_html = f"""
        <div class="rec-card" style="animation-delay:{delay:.2f}s">
            {poster_html}
            <div class="rec-body">
                <div class="rec-rank">#{rank+1:02d} &nbsp;·&nbsp; {release_yr}</div>
                <div class="rec-title">{title}</div>
                <div class="rec-meta">
                    {genre_tags}
                    <span class="pill pill-yellow" style="font-size:9px;padding:2px 7px;">⚡ {pct}%</span>
                    {'<span class="pill" style="font-size:9px;padding:2px 7px;">⭐ '+ f"{vote:.1f}"+'</span>' if vote else ""}
                </div>
                {'<div class="rec-cast"><span>Cast:</span><div class="cast-row">' + cast_chips + '</div></div>' if cast_chips else ""}
            </div>
        </div>
        """

        with cols[rank % 5]:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown(
        '<p class="foot">Content-Based Filtering &nbsp;·&nbsp; TF-IDF + Cosine Similarity &nbsp;·&nbsp; TMDB API</p>',
        unsafe_allow_html=True
    )
