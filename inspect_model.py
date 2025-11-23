import pickle
import sys

model_path = "models/xgboost_stress_model.pkl"

try:
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        # Check if model is inside
        for k, v in data.items():
            print(f"Key '{k}' type: {type(v)}")
    else:
        print(f"Dir: {dir(data)}")

except Exception as e:
    print(f"Error loading pickle: {e}")
