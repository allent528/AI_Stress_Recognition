# FILE: ai_engine/engine.py
from typing import List
import pandas as pd
from .types import Model, PredictionInput, PredictionOutput

class AIEngine:
    """
    Core AI Engine for running predictions using a loaded model.
    """
    def __init__(self, model: Model):
        """
        Initialize the AI Engine with a trained model.

        Args:
            model (Model): The loaded machine learning model.
        """
        self.model = model
        print(f"DEBUG: Engine initialized with model type: {type(self.model)}")
        print(f"DEBUG: Model dir: {dir(self.model)}")

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Run a single prediction on user-provided physiological data.

        Args:
            input_data (dict): Dictionary containing input features (e.g., {'hr': 70, 'hrv': 50}).

        Returns:
            dict: Prediction result containing 'stress_prob' and 'is_stressed'.
        """
        try:
            # Convert input dict to DataFrame for model prediction (expects 2D array-like)
            df = pd.DataFrame([input_data])
            
            # Get probability of positive class (stress)
            # Assuming binary classification where class 1 is 'stressed'
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(df)
                stress_prob = float(probs[0][1]) # Probability of class 1
            else:
                # Fallback: use predict() and assign 1.0 or 0.0
                prediction = self.model.predict(df)[0]
                stress_prob = 1.0 if prediction == 1 else 0.0

            is_stressed = stress_prob > 0.5

            return {
                "stress_prob": stress_prob,
                "is_stressed": is_stressed
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def batch_predict(self, input_list: List[PredictionInput]) -> List[PredictionOutput]:
        """
        Run predictions on a batch of inputs.

        Args:
            input_list (list): List of input dictionaries.

        Returns:
            list: List of prediction result dictionaries.
        """
        results = []
        for item in input_list:
            results.append(self.predict(item))
        return results
