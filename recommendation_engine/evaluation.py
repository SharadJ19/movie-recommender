# recommendation_engine/evaluation.py
"""
evaluation.py

Evaluation module for Collaborative Filtering:
1. Train-test split for user-item ratings (dtype-safe)
2. RMSE computation for model performance
3. Vectorized predictions for speed
4. Supports both user-based and item-based CF

Author: Sharad Chandel
Date: 2025-08-30
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from recommendation_engine.collaborative import CollaborativeFiltering

class Evaluator:
    def __init__(self, user_item_matrix: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
        """
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix with ratings
            test_ratio (float): Fraction of ratings to hold out for testing
            random_state (int): Seed for reproducibility
        """
        # Ensure float dtype to avoid dtype warnings
        self.user_item_matrix = user_item_matrix.astype(float).copy()
        self.train_matrix = self.user_item_matrix.copy()
        self.test_matrix = pd.DataFrame(0.0, index=user_item_matrix.index, columns=user_item_matrix.columns)
        self.test_ratio = test_ratio
        self.random_state = random_state

    # ------------------- Train-Test Split -------------------
    def train_test_split(self):
        """Randomly mask test_ratio of ratings for each user (dtype-safe)"""
        np.random.seed(self.random_state)
        for user in self.user_item_matrix.index:
            rated_items = self.user_item_matrix.loc[user][self.user_item_matrix.loc[user] > 0].index
            n_test = max(1, int(len(rated_items) * self.test_ratio))
            test_items = np.random.choice(rated_items, size=n_test, replace=False)
            self.train_matrix.loc[user, test_items] = 0.0
            self.test_matrix.loc[user, test_items] = self.user_item_matrix.loc[user, test_items].astype(float)
        return self.train_matrix, self.test_matrix

    # ------------------- Vectorized Prediction -------------------
    def predict_all(self, cf_model: CollaborativeFiltering, method: str = 'user') -> pd.DataFrame:
        """
        Vectorized prediction for all users to avoid slow nested loops.
        
        Args:
            cf_model (CollaborativeFiltering): CF model initialized with train_matrix
            method (str): 'user' or 'item'
        
        Returns:
            pd.DataFrame: Predicted ratings for all users
        """
        predictions = pd.DataFrame(0.0, index=self.train_matrix.index, columns=self.train_matrix.columns, dtype=float)

        if method == 'user':
            if cf_model.user_similarity is None:
                cf_model.compute_user_similarity()
            similarity_matrix = cf_model.user_similarity.copy()
            np.fill_diagonal(similarity_matrix.values, 0)  # exclude self
            sim_sums = similarity_matrix.sum(axis=1) + 1e-8
            weighted_ratings = self.train_matrix.T.dot(similarity_matrix.T) / sim_sums
            weighted_ratings = weighted_ratings.T
            # Mask already rated items
            predictions = weighted_ratings.where(self.train_matrix == 0, 0.0)

        elif method == 'item':
            if cf_model.item_similarity is None:
                cf_model.compute_item_similarity()
            pred = cf_model.item_similarity.dot(self.train_matrix.T) / (
                np.abs(cf_model.item_similarity).sum(axis=1).values[:, None] + 1e-8
            )
            predictions = pred.T
            predictions = predictions.where(self.train_matrix == 0, 0.0)

        else:
            raise ValueError("method must be 'user' or 'item'")

        return predictions

    # ------------------- RMSE Computation -------------------
    def compute_rmse(self, predictions: pd.DataFrame) -> float:
        """
        Compute RMSE between predictions and test_matrix
        
        Returns:
            float: RMSE score
        """
        mask = self.test_matrix > 0
        y_true = self.test_matrix[mask]
        y_pred = predictions[mask]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse


# ------------------- Example Walkthrough -------------------
if __name__ == "__main__":
    from recommendation_engine.data_loader import DataLoader
    from recommendation_engine.collaborative import CollaborativeFiltering

    # Load user-item matrix
    loader = DataLoader(dataset_path="ml-latest-small")
    loader.load_ratings()
    user_item_matrix = loader.get_user_item_matrix()

    # Train-test split
    evaluator = Evaluator(user_item_matrix=user_item_matrix, test_ratio=0.2)
    train_matrix, test_matrix = evaluator.train_test_split()
    print("Train/Test split done.")

    # Initialize CF model on train_matrix
    cf = CollaborativeFiltering(train_matrix)
    cf.compute_user_similarity()
    cf.compute_item_similarity()

    # Predict ratings (vectorized, failsafe)
    predictions_user = evaluator.predict_all(cf_model=cf, method='user')
    predictions_item = evaluator.predict_all(cf_model=cf, method='item')

    # Compute RMSE
    rmse_user = evaluator.compute_rmse(predictions_user)
    rmse_item = evaluator.compute_rmse(predictions_item)

    print(f"User-based CF RMSE: {rmse_user:.4f}")
    print(f"Item-based CF RMSE: {rmse_item:.4f}")
