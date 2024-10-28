# import pandas as pd
# import streamlit as st
# from utils import load_data
# from recommender import load_knn_model, get_song_features
# from sklearn.preprocessing import StandardScaler

# # Load the data and preprocess
# data = pd.read_csv("data/songs.csv")
# numeric_features = data[['rating', 'duration_ms']].select_dtypes(include=['float64', 'int64'])
# scaler = StandardScaler().fit(numeric_features)  # Fit the scaler on numeric features

# # Load the pre-trained KNN model
# knn_model = load_knn_model()

# st.title("Music Recommendation System")

# # User input for song title
# title = st.text_input("Enter a Song Title")

# if st.button("Get Recommendations"):
#     # Get features for the input song
#     song_features = get_song_features(title, data, scaler)
    
#     if song_features is not None:
#         # Find similar songs using KNN
#         distances, indices = knn_model.kneighbors(song_features, n_neighbors=10)
        
#         # Retrieve recommendations
#         recommended_songs = data.iloc[indices.flatten()]

#         # Display recommendations
#         st.write("Recommended Songs:")
#         for index, row in recommended_songs.iterrows():
#             st.write(f"- {row['title']} by {row['artist_name']}")
#     else:
#         st.write("Song not found. Please enter a valid song title.")


import streamlit as st
from utils import load_data
from recommender import load_model_and_scaler, get_song_features, recommend_songs

# Load data, model, and scaler
data, numeric_features = load_data("data/songs.csv")
knn_model, scaler = load_model_and_scaler()

st.title("Music Recommendation System")

# User input for song title
title = st.text_input("Enter a Song Title")

if st.button("Get Recommendations"):
    song_features = get_song_features(title, data, scaler)
    
    if song_features is not None:
        recommendations = recommend_songs(song_features, data, knn_model)
        st.write("Recommended Songs:")
        for index, row in recommendations.iterrows():
            st.write(f"- {row['title']} by {row['artist_name']}")
    else:
        st.write("Song not found. Please enter a valid song title.")
