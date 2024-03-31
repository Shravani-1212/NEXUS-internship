import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from sklearn.ensemble import RandomForestRegressor

# Set up Spotify API
scope = 'playlist-modify-private'
sp = Spotify(auth_manager=SpotifyOAuth(scope=scope))

# Define functions
def get_song_features(song_uri):
    """Get song features using Spotify Web API."""
    features = sp.audio_features(song_uri)[0]
    return features

def get_user_features(user_id):
    """Get user's listening history using Spotify Web API."""
    results = sp.current_user_top_artists(limit=10)
    top_artists = results['items']
    artist_ids = [artist['id'] for artist in top_artists]
    artist_uris = ['spotify:artist:' + artist_id for artist_id in artist_ids]
    artist_features = sp.artists(artist_uris)
    artist_features = [feature for artist in artist_features for feature in artist['audio_features']]
    artist_features = pd.DataFrame(artist_features)
    artist_features = artist_features.dropna()
    return artist_features

# Rest of your code (Please include the missing parts)

# Example usage
# user_id = sp.me()['id']
# user_features = get_user_features(user_id)
# song_uri = 'spotify:track:6rqhFgbbKwnb9MLmUQDhG6'
# song_features = get_song_features(song_uri)


def get_similarity_matrix(features):
    """Calculate similarity matrix using cosine distance."""
    features = features.drop(['danceability', 'energy', 'loudness', 'speechiness'], axis=1)
    similarity_matrix = 1 - distance.cdist(features, features, 'cosine')
    return similarity_matrix

def get_recommendations(similarity_matrix, features):
    """Get song recommendations based on collaborative filtering."""
    n = len(features)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_index = indices[:100]
    train_index = indices[100:]
    train_features = features.iloc[train_index]
    test_features = features.iloc[test_index]
    train_labels = similarity_matrix[train_index][:, train_index]
    test_labels = similarity_matrix[test_index][:, train_index]
    k = 10
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_labels)
    distances, indices = nn.kneighbors(test_labels)
    recommendations = []
    for i in range(len(test_features)):
        recommendations.extend(train_features.iloc[indices[i]].index.tolist())
    recommendations = list(set(recommendations))
    return recommendations

def create_playlist(name, description):
    """Create a new playlist using Spotify Web API."""
    results = sp.user_playlist_create(sp.me()['id'], name, description=description)
    return results['id']

def add_songs_to_playlist(playlist_id, songs):
    """Add songs to a playlist using Spotify Web API."""
    uris = ['spotify:track:' + song for song in songs]
    results = sp.playlist_add_items(playlist_id, uris)
    return results

# Get user features
user_id = sp.me()['id']
user_features = get_user_features(user_id)

# Get song features
song_features = pd.DataFrame()
for uri in ['spotify:track:6rqhFgbbKwnb9MLmUQDhG6', 'spotify:track:5xWtjybIUYsMnFbIKf5Qa0']:
    features = get_song_features(uri)
    song_features = pd.concat([song_features, pd.DataFrame([features])])
