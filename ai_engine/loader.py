# FILE: ai_engine/loader.py
import os
import pickle
import joblib
from .types import Model

def load_model(path: str) -> Model:
    """
    Loads a machine learning model from a .pkl or .joblib file.

    Args:
        path (str): The file path to the model.

    Returns:
        Model: The loaded model object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        if ext == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
                # Handle case where model is wrapped in a dict (e.g. {'model': ..., 'scaler': ...})
                if isinstance(data, dict) and 'model' in data:
                    return data['model']
                return data
        elif ext == '.joblib':
            return joblib.load(path)
        else:
            raise ValueError(f"Unsupported model format: {ext}. Expected .pkl or .joblib")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")
