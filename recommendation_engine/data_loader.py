# recommendation_engine/data_loader.py
"""
data_loader.py

Responsible for:
1. Loading MovieLens CSV files: movies.csv, ratings.csv, tags.csv, links.csv
2. Basic preprocessing
3. Preparing user-item matrices for collaborative filtering

Author: Sharad Chandel
Date: 2025-08-30
"""

import pandas as pd
import numpy as np
from typing import Tuple

class DataLoader:
    def __init__(self, dataset_path: str):
        """
        Initialize the DataLoader with the path to MovieLens CSV files.
        
        Args:
            dataset_path (str): Path to the folder containing the CSV files.
        """
        self.dataset_path = dataset_path
        self.movies = None
        self.ratings = None
        self.tags = None
        self.links = None

    def load_movies(self) -> pd.DataFrame:
        """Load movies.csv with basic preprocessing"""
        path = f"{self.dataset_path}/movies.csv"
        self.movies = pd.read_csv(path)
        # Optional: split genres into list for future filtering
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        return self.movies

    def load_ratings(self) -> pd.DataFrame:
        """Load ratings.csv"""
        path = f"{self.dataset_path}/ratings.csv"
        self.ratings = pd.read_csv(path)
        # Ensure ratings are float
        self.ratings['rating'] = self.ratings['rating'].astype(float)
        return self.ratings

    def load_tags(self) -> pd.DataFrame:
        """Load tags.csv"""
        path = f"{self.dataset_path}/tags.csv"
        self.tags = pd.read_csv(path)
        return self.tags

    def load_links(self) -> pd.DataFrame:
        """Load links.csv"""
        path = f"{self.dataset_path}/links.csv"
        self.links = pd.read_csv(path)
        return self.links

    def get_user_item_matrix(self) -> pd.DataFrame:
        """
        Prepare user-item rating matrix for collaborative filtering.
        Users are rows, movies are columns, ratings are values.
        Missing ratings are filled with 0 (sparse approach).
        """
        if self.ratings is None:
            raise ValueError("Ratings data not loaded. Call load_ratings() first.")
        user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        return user_item_matrix

# --- Example Walkthrough ---

if __name__ == "__main__":
    loader = DataLoader(dataset_path="ml-latest-small")  # replace with your dataset folder
    movies = loader.load_movies()
    ratings = loader.load_ratings()
    tags = loader.load_tags()
    links = loader.load_links()
    
    print("Movies sample:\n", movies.head())
    print("Ratings sample:\n", ratings.head())
    
    user_item_matrix = loader.get_user_item_matrix()
    print("User-Item Matrix shape:", user_item_matrix.shape)
    print("Sample matrix:\n", user_item_matrix.head())
