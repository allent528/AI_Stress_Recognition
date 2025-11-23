# How to Use Your XGBoost Model

I noticed your `AI_Proj_phase3.ipynb` trains an XGBoost model but doesn't save it to a file. To use it with this system, follow these steps:

## 1. Save Your Model
Open `AI_Proj_phase3.ipynb`, run the cells to train `xgb_model`, and then run this **new cell** at the end:

```python
import pickle
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the trained XGBoost model
model_path = "models/xgboost_stress_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(xgb_model, f)

print(f"Model saved to {model_path}")
```

## 2. Update `main.py`
Modify `main.py` to load your new model and use the correct features.

```python
# In main.py

def main():
    # 1. Update Model Path
    MODEL_PATH = os.path.join("models", "xgboost_stress_model.pkl") 
    
    # ... load model ...

    # 2. Update Input Features
    # Your notebook uses 88 features (ACC_axis0_mean, etc.)
    # You need to provide a dictionary with ALL these features.
    # For testing, you might want to grab a row from your test set:
    
    # Example (conceptual):
    # user_input = {
    #     "ACC_axis0_mean": 0.5,
    #     "ACC_axis0_std": 0.1,
    #     ... (all 88 features)
    # }
    
    # ... rest of script ...
```

> **Note**: Since your XGBoost model expects 88 specific features (scaled), you'll need to ensure the input dictionary in `main.py` matches exactly what the model expects, including any preprocessing (StandardScaler) you did in the notebook. You might need to save and load your `scaler` as well!

## 3. Saving the Scaler (Recommended)
In your notebook:
```python
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

In `main.py`:
```python
import pandas as pd

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Preprocess input
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = engine.predict(input_scaled) # You might need to adjust Engine to handle array input if needed
```
