import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv("movie_dataset.csv")
    df['overview'] = df['overview'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

df, cosine_sim = load_data()

st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox(
    "Select a Movie",
    df['title'].values
)

if st.button("Recommend"):
    movie_index = df[df['title'] == movie_name].index[0]

    distances = cosine_sim[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    st.subheader("Recommended Movies")

    for movie in movies_list:
        st.write(df.iloc[movie[0]].title)
