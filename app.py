from flask import Flask, render_template, request
from recommendation_engine.data_loader import DataLoader
from recommendation_engine.collaborative import CollaborativeFiltering

app = Flask(__name__)

# ------------------- Load Data -------------------
DATASET_PATH = "ml-latest-small"
loader = DataLoader(dataset_path=DATASET_PATH)
loader.load_movies()
loader.load_ratings()
loader.load_links()  # optional, ignored in this version
user_item_matrix = loader.get_user_item_matrix()

cf_model = CollaborativeFiltering(user_item_matrix)
cf_model.compute_user_similarity()
cf_model.compute_item_similarity()

# ------------------- Routes -------------------

@app.route('/')
def home():
    """Home page with Most Popular Movies and interactive form"""
    # Most popular movies: consider only movies with >= 50 ratings
    min_ratings = 50
    ratings_summary = loader.ratings.groupby('movieId')['rating'].agg(['mean','count'])
    popular_movies_df = ratings_summary[ratings_summary['count'] >= min_ratings]
    popular_movies_df = popular_movies_df.sort_values('mean', ascending=False).head(10)
    
    popular_movies = []
    for movie_id, row in popular_movies_df.iterrows():
        title = loader.movies.loc[loader.movies['movieId'] == movie_id, 'title'].values[0]
        avg_rating = round(row['mean'], 2)
        popular_movies.append((title, avg_rating))
    
    return render_template("index.html", max_user=max(user_item_matrix.index), popular_movies=popular_movies)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    """Show top-N recommendations for selected user, method, and optional genre"""
    if request.method == 'POST':
        user_id = int(request.form.get('user_id', 1))
        method = request.form.get('method', 'user')
        top_n = int(request.form.get('top_n', 10))
        genre_filter = request.form.get('genre', '').lower()
    else:
        user_id = 1
        method = 'user'
        top_n = 10
        genre_filter = ''

    try:
        if method == 'user':
            recs = cf_model.recommend_user_based(user_id=user_id, top_n=user_item_matrix.shape[1])
        else:
            recs = cf_model.recommend_item_based(user_id=user_id, top_n=user_item_matrix.shape[1])
    except KeyError:
        recs = []

    # Filter already rated movies
    rated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recs = [(movie_id, rating) for movie_id, rating in recs if movie_id not in rated_movies]

    # Optional genre filter
    if genre_filter:
        recs = [(movie_id, rating) for movie_id, rating in recs 
                if genre_filter in [g.lower() for g in loader.movies.loc[loader.movies['movieId'] == movie_id, 'genres'].values[0]]]

    # Take top-N
    recs = recs[:top_n]

    # Map to titles and average ratings
    recommendations_list = []
    for movie_id, rating in recs:
        title = loader.movies.loc[loader.movies['movieId'] == movie_id, 'title'].values[0]
        avg_rating_row = loader.ratings[loader.ratings['movieId'] == movie_id]['rating']
        avg_rating = round(avg_rating_row.mean(), 2) if not avg_rating_row.empty else None
        recommendations_list.append((title, rating, avg_rating))

    return render_template("recommendations.html",
                           user_id=user_id,
                           method=method,
                           recommendations=recommendations_list,
                           genre_filter=genre_filter)

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT
    app.run(host="0.0.0.0", port=port)