"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

# This code is a Python module for a music recommendation system that uses the implicit library to generate recommendations based on user interactions with artists.
# Path from pathlib: Handles file paths.
# Tuple and List from typing: Used for type hints.
# implicit: Library for collaborative filtering recommendations.
# scipy: Provides sparse matrix support.
# load_user_artists and ArtistRetriever: Functions and classes imported from the MR module 
from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from MR import load_user_artists, ArtistRetriever

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

# Purpose: Computes music recommendations using a collaborative filtering model from the implicit library.
# Constructor: Initializes the recommender with an ArtistRetriever instance and an implicit model.
    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

# fit Method: Trains the model using the user-artist interaction matrix.
    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

# recommend Method: Generates recommendations for a given user.
# Parameters:
# user_id: ID of the user for whom recommendations are generated.
# user_artists_matrix: Sparse matrix of user-artist interactions.
# n: Number of recommendations to return.
# Returns: A tuple of two lists:
# artists: Names of the recommended artists.
# scores: Corresponding recommendation scores.
    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

# Load Data:

# user_artists: Loads the user-artist interaction matrix from the specified file path.
# artist_retriever: Instantiates and loads artist data.
# Model Setup:

# implict_model: Creates an Alternating Least Squares (ALS) model with specific parameters for collaborative filtering.
# Recommendation:

# recommender: Creates an instance of ImplicitRecommender, fits it with the user-artists matrix, and generates recommendations for user ID 2.
# Output:

# Prints the top 5 recommended artists and their corresponding scores.
if __name__ == "__main__":

    # load user artists matrix
    user_artists = load_user_artists(Path("C:\\Users\\singh\\OneDrive\\Desktop\\Music Recommendation System\\hetrec2011-lastfm-2k\\user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("C:\\Users\\singh\\OneDrive\\Desktop\\Music Recommendation System\\hetrec2011-lastfm-2k\\artists.dat"))

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implict_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(3, user_artists, n=10)

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")

# Artist Name: Name of the recommended artist.
# Score: The recommendation score for that artist, which indicates the strength of the recommendation.