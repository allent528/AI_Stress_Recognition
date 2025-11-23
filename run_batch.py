import os
import sys
import pandas as pd
import pickle
from ai_engine import AIEngine, generate_historical_feedback, load_model

def run_batch_processing():
    # Configuration
    INPUT_CSV = os.path.join("data", "processed_user_data.csv")
    MODEL_PATH = os.path.join("models", "xgboost_stress_model.pkl") # Default to XGBoost model if available
    # MODEL_PATH = os.path.join("models", "my_model.pkl") # Force dummy model for verification
    SCALER_PATH = os.path.join("models", "scaler.pkl")
    OUTPUT_CSV = os.path.join("data", "outputs", "batch_predictions.csv")

    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        print("Please export your data from the notebook using EXPORT_DATA_INSTRUCTIONS.md")
        sys.exit(1)

    # 1. Load Data
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'userId' not in df.columns:
        print("Error: 'userId' column missing in input CSV.")
        sys.exit(1)

    # 2. Load Model and Scaler
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to dummy model if XGBoost not found (for testing)
        fallback_path = os.path.join("models", "my_model.pkl")
        if os.path.exists(fallback_path):
            print(f"Falling back to dummy model at {fallback_path}")
            model = load_model(fallback_path)
            SCALER_PATH = None # Dummy model might not need the specific scaler
        else:
            print("No model found. Please train a model first.")
            sys.exit(1)

    print(f"Loaded model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")


    scaler = None
    if SCALER_PATH and os.path.exists(SCALER_PATH):
        print(f"Loading scaler from {SCALER_PATH}...")
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading scaler: {e}")

    # 3. Initialize Engine
    engine = AIEngine(model)

    # 4. Process by User
    user_ids = df['userId'].unique()
    print(f"Found {len(user_ids)} unique users.")

    all_results = []

    for uid in user_ids:
        print(f"Processing User {uid}...")
        user_df = df[df['userId'] == uid]
        
        # Prepare features (drop userId)
        features_df = user_df.drop(columns=['userId'])
        
        # Scale if necessary
        if scaler:
            try:
                # Ensure columns match scaler's expected features
                # This might fail if features don't match exactly, so we add a check
                features_scaled = scaler.transform(features_df)
                # Convert back to list of dicts for engine (or modify engine to accept array)
                # Engine expects dict, but we can optimize. 
                # Actually, Engine.predict expects dict. batch_predict expects list of dicts.
                # Let's convert scaled array back to DataFrame/dict
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            except Exception as e:
                print(f"Scaling failed for user {uid}: {e}")
                continue

        # Convert to list of dicts for batch_predict
        input_list = features_df.to_dict(orient='records')
        
        # Run Predictions
        try:
            predictions = engine.batch_predict(input_list)
        except Exception as e:
            print(f"Prediction failed for user {uid}: {e}")
            continue

        # Aggregate Results
        stress_probs = [p['stress_prob'] for p in predictions]
        is_stressed_list = [p['is_stressed'] for p in predictions]
        
        avg_stress = sum(stress_probs) / len(stress_probs) if stress_probs else 0
        stress_freq = sum(is_stressed_list) / len(is_stressed_list) if is_stressed_list else 0
        
        # Generate Historical Feedback
        history_summary = {
            "avg_stress_prob": avg_stress,
            "stress_frequency": stress_freq
        }
        feedback = generate_historical_feedback(history_summary)
        
        print(f"  -> Avg Stress: {avg_stress:.2f}, Freq: {stress_freq:.2f}")
        print(f"  -> Feedback: {feedback[:50]}...")

        # Store Result
        all_results.append({
            "userId": uid,
            "avg_stress_prob": avg_stress,
            "stress_frequency": stress_freq,
            "feedback": feedback
        })

    # 5. Save Batch Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nBatch processing complete. Results saved to {OUTPUT_CSV}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_batch_processing()
