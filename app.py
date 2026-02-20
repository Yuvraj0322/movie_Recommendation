# app.py
import streamlit as st
import pickle

# ----------------- Load Data -----------------
movies = pickle.load(open('artifacts/movies.pkl', 'rb'))
similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))

# ----------------- Recommendation Function -----------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# ----------------- Streamlit UI -----------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Type or select a movie",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(movie)