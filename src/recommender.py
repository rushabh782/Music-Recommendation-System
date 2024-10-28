# # src/recommender.py
# import pickle
# import pandas as pd
# # from sklearn.neighbors import NearestNeighbors

# def load_knn_model():
#     # Load the KNN model from the pickle file
#     with open('models/knn_model.pkl', 'rb') as model_file:
#         knn_model = pickle.load(model_file)
#     return knn_model

# def get_recommendations(title, song_features, knn_model, data, n_recommendations=5):
#     """Generate song recommendations based on a given song name."""
#     song_idx = data.index[data['title'] == title].tolist()
    
#     if not song_idx:
#         return pd.DataFrame()  # Return empty DataFrame if song not found
    
#     song_idx = song_idx[0]

#     distances, indices = knn_model.kneighbors([song_features.iloc[song_idx]], n_neighbors=n_recommendations + 1)
#     recommended_songs = data.iloc[indices.flatten()[1:]]  # Skip the input song itself
#     return recommended_songs[['title', 'artist_name']]

# def get_song_features(song_name, data, scaler):
#     # Clean the input song name
#     cleaned_song_name = song_name.strip().lower()

#     # Find the song in the dataset
#     song = data[data['title'].str.lower() == cleaned_song_name]
    
#     if song.empty:
#         return None  # Return None if song not found

#     # Extract relevant features for the found song
#     song_features = song[['rating', 'duration_ms']].select_dtypes(include=['float64', 'int64']).values

#     # Check if song_features has the correct shape before scaling
#     if song_features.shape[1] != 2:
#         return None  # Ensure the correct number of features

#     # Scale the features using the same scaler used for training
#     scaled_features = scaler.transform(song_features)
    
#     return scaled_features  # Return scaled features



import pickle
import pandas as pd

def load_model_and_scaler():
    with open('models/knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)
    with open('models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return knn_model, scaler

def get_song_features(title, data, scaler):
    title = title.strip().lower()
    song = data[data['title'].str.lower() == title]
    
    if not song.empty:
        song_features = song[['rating', 'duration_ms']].values
    else:
        song_features = [[3.0, 200000]]
    
    scaled_features = scaler.transform(song_features)
    return scaled_features

def recommend_songs(song_features, data, knn_model):
    distances, indices = knn_model.kneighbors(song_features, n_neighbors=10)
    recommended_songs = data.iloc[indices.flatten()]
    return recommended_songs[['title', 'artist_name']]
