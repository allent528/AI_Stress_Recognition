# FILE: main.py
import os
import sys
from ai_engine import load_model, AIEngine, CSVStorage, generate_feedback

def main():
    # Configuration
    MODEL_PATH = os.path.join("models", "xgboost_stress_model.pkl")
    # MODEL_PATH = os.path.join("models", "my_model.pkl") # Use dummy model for verification
    OUTPUT_CSV = os.path.join("data", "outputs", "predictions.csv")
    
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure a trained model exists at the specified path.")
        sys.exit(1)
    
    # 2. Initialize Engine
    engine = AIEngine(model)
    
    # 3. Define Example Input
    # Physiological data: Heart Rate (bpm), HRV (ms), ECG feature (arbitrary unit)
    user_input = {
        "hr": 85,
        "hrv": 45,
        "ecg_feature": 0.12
    }
    print(f"Processing input: {user_input}")
    
    # 4. Run Prediction
    try:
        prediction = engine.predict(user_input)
        print(f"Prediction result: {prediction}")
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)
    
    # 5. Generate Feedback
    print("Generating feedback...")
    
    # Load history for context
    history_stats = {}
    if os.path.exists(OUTPUT_CSV):
        try:
            import pandas as pd
            history_df = pd.read_csv(OUTPUT_CSV)
            if not history_df.empty and 'is_stressed' in history_df.columns:
                # Calculate recent stress frequency (last 10 records)
                recent_records = history_df.tail(10)
                stress_count = recent_records['is_stressed'].sum()
                total_count = len(recent_records)
                history_stats['recent_stress_freq'] = stress_count / total_count if total_count > 0 else 0.0
                print(f"Historical context: Recent stress freq = {history_stats['recent_stress_freq']:.2f}")
        except Exception as e:
            print(f"Warning: Could not load history: {e}")

    feedback = generate_feedback(prediction, history_stats)
    print(f"Feedback: {feedback}")
    
    # 6. Save Results
    storage = CSVStorage(OUTPUT_CSV)
    
    # Combine all data for storage
    record = {**user_input, **prediction, "feedback": feedback}
    
    try:
        storage.save_record(record)
        print(f"Results saved to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    print("Workflow completed successfully.")

if __name__ == "__main__":
    main()
