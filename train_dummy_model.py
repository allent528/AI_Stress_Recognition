# FILE: train_dummy_model.py
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_dummy_model():
    # Create dummy data
    # Features: hr, hrv, ecg_feature
    # Target: is_stressed (0 or 1)
    data = {
        'hr': [60, 100, 70, 110, 65, 95],
        'hrv': [60, 20, 55, 15, 50, 25],
        'ecg_feature': [0.05, 0.2, 0.06, 0.25, 0.04, 0.18],
        'is_stressed': [0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    X = df[['hr', 'hrv', 'ecg_feature']]
    y = df['is_stressed']
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "my_model.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Dummy model saved to {model_path}")

if __name__ == "__main__":
    train_dummy_model()
