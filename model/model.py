# model.py
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load Datasets ----------------
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Keep necessary columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Drop rows with null values
movies.dropna(inplace=True)

# ----------------- Helper Functions -----------------
def convert(text):
    """Convert JSON column to list of names"""
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def fetch_cast(text):
    """Get top 3 actors"""
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    """Get director name"""
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# ----------------- Process Columns -----------------
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(fetch_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces in multi-word names
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# ----------------- Create Tags -----------------
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id','title','tags']]

# Convert list of tags to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

# ----------------- Vectorization -----------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# ----------------- Cosine Similarity -----------------
similarity = cosine_similarity(vectors)

# ----------------- Save Pickle Files -----------------
pickle.dump(new_df, open('artifacts/movies.pkl','wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl','wb'))

print("✅ movies.pkl and similarity.pkl created successfully!")