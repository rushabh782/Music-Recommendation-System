import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    numeric_features = data[['rating', 'duration_ms']].select_dtypes(include=['float64', 'int64'])
    return data, numeric_features

def preprocess_data(numeric_features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    return scaled_features, scaler
