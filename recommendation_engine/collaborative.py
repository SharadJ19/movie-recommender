# recommendation_engine/collaborative.py
"""
collaborative.py

Collaborative Filtering Algorithms:
1. User-based CF
2. Item-based CF
3. Top-N recommendations
4. Uses cosine similarity from scikit-learn
5. Fixed alignment issues for Pandas dot products

Author: Sharad Chandel
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class CollaborativeFiltering:
    def __init__(self, user_item_matrix: pd.DataFrame):
        """
        Args:
            user_item_matrix (pd.DataFrame): Users as rows, movies as columns, ratings as values.
        """
        self.user_item_matrix = user_item_matrix
        self.user_similarity = None
        self.item_similarity = None

    # ------------------- Similarity Computation -------------------
    def compute_user_similarity(self):
        """Compute cosine similarity between users"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(self.user_similarity, 
                                            index=self.user_item_matrix.index, 
                                            columns=self.user_item_matrix.index)
        return self.user_similarity

    def compute_item_similarity(self):
        """Compute cosine similarity between items (movies)"""
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(self.item_similarity, 
                                            index=self.user_item_matrix.columns, 
                                            columns=self.user_item_matrix.columns)
        return self.item_similarity

    # ------------------- Recommendation Generation -------------------
    def recommend_user_based(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend movies for a given user using user-based CF.
        
        Args:
            user_id (int): ID of the user
            top_n (int): Number of recommendations
        
        Returns:
            List of (movieId, predicted_rating) tuples sorted by rating descending
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        similar_users = self.user_similarity[user_id]
        
        # Exclude self
        similar_users = similar_users.drop(user_id)
        
        # Align similar_users with user_item_matrix rows
        similar_users_aligned = similar_users.reindex(self.user_item_matrix.index).fillna(0)
        
        # Weighted sum of ratings from similar users
        weighted_ratings = self.user_item_matrix.T.dot(similar_users_aligned)
        similarity_sums = similar_users_aligned.sum()
        
        # Avoid division by zero
        predicted_ratings = weighted_ratings / (similarity_sums + 1e-8)
        
        # Remove already rated movies
        predicted_ratings = predicted_ratings[user_ratings == 0]
        
        # Return top-N recommendations
        top_movies = predicted_ratings.sort_values(ascending=False).head(top_n)
        return list(top_movies.items())

    def recommend_item_based(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend movies for a given user using item-based CF.
        
        Args:
            user_id (int): ID of the user
            top_n (int): Number of recommendations
        
        Returns:
            List of (movieId, predicted_rating) tuples sorted by rating descending
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Align user ratings with item similarity columns
        user_ratings_aligned = user_ratings.reindex(self.user_item_matrix.columns).fillna(0)
        
        # Predicted rating = weighted sum of item similarities * user ratings
        pred_ratings = self.item_similarity.dot(user_ratings_aligned) / (np.abs(self.item_similarity).sum(axis=1) + 1e-8)
        
        # Remove already rated movies
        pred_ratings = pred_ratings[user_ratings == 0]
        
        # Return top-N
        top_movies = pred_ratings.sort_values(ascending=False).head(top_n)
        return list(top_movies.items())

# ------------------- Complexity Notes -------------------
"""
Time Complexity:
- compute_user_similarity: O(U^2 * M) where U = #users, M = #movies
- compute_item_similarity: O(M^2 * U)
- recommend_user_based: O(U*M)
- recommend_item_based: O(M^2)

Space Complexity:
- Similarity matrices: O(U^2) or O(M^2)
- Predictions: O(M)
"""

# ------------------- Example Walkthrough -------------------
if __name__ == "__main__":
    from recommendation_engine.data_loader import DataLoader
    
    # Load data
    loader = DataLoader(dataset_path="ml-latest-small")
    loader.load_ratings()
    user_item_matrix = loader.get_user_item_matrix()
    
    cf = CollaborativeFiltering(user_item_matrix)
    
    print("Computing user similarity...")
    cf.compute_user_similarity()
    print("User similarity matrix shape:", cf.user_similarity.shape)
    
    print("Generating top 5 recommendations for userId=1 (user-based)...")
    recs_user = cf.recommend_user_based(user_id=1, top_n=5)
    print(recs_user)
    
    print("Generating top 5 recommendations for userId=1 (item-based)...")
    recs_item = cf.recommend_item_based(user_id=1, top_n=5)
    print(recs_item)
