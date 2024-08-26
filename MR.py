"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""
# This code provides a module for handling data related to collaborative filtering algorithms in a music recommendation system. It includes functionality for loading user-artist interactions and artist information, as well as for retrieving artist names based on their IDs.


from pathlib import Path

import scipy
import pandas as pd

# Purpose: Loads user-artist interaction data and returns a sparse matrix.
# Parameters:
# user_artists_file (Path): Path to the file containing user-artist interactions.
# Steps:
# Read the file into a DataFrame using pandas.read_csv.
# Set userID and artistID as a multi-level index.
# Create a COO (Coordinate) format sparse matrix from the DataFrame.
# Convert the COO matrix to CSR (Compressed Sparse Row) format for efficient row slicing.
# Return: A CSR matrix representing user-artist interactions.
def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user artists file and return a user-artists matrix in csr
    fromat.
    """
    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()

# Purpose: Handles artist data, including loading artist information and retrieving artist names.
# Methods:
# __init__: Initializes an instance with a private attribute _artists_df for storing artist data.
# get_artist_name_from_id: Retrieves the artist's name based on the artist ID.
# load_artists: Loads artist information from a file and stores it as a DataFrame with id as the index.
class ArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID."""

    def __init__(self):
        self._artists_df = None

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """Return the artist name from the artist ID."""
        return self._artists_df.loc[artist_id, "name"]

    def load_artists(self, artists_file: Path) -> None:
        """Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file ,sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df

# Purpose: This block is executed when the script is run directly (not imported as a module).
# Steps:
# Load the user-artist matrix from a specified file and print it.
# Instantiate ArtistRetriever, load artist data from a specified file, and retrieve the name of the artist with ID 1, then print it.
if __name__ == "__main__":
    user_artists_matrix = load_user_artists(
        Path("C:\\Users\\singh\\OneDrive\\Desktop\\Music Recommendation System\\hetrec2011-lastfm-2k\\user_artists.dat")
    )
    print(user_artists_matrix)

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("C:\\Users\\singh\\OneDrive\\Desktop\\Music Recommendation System\\hetrec2011-lastfm-2k\\artists.dat"))
    artist = artist_retriever.get_artist_name_from_id(1)
    print(artist)