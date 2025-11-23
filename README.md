# Python AI System

A modular system that integrates a trained ML model, CSV-based data storage, and an LLM for personalized text output.

## Structure

- `ai_engine/`: Core logic for loading models, running predictions, storage, and LLM feedback.
- `models/`: Directory for trained ML models (`.pkl` or `.joblib`).
- `data/outputs/`: Directory where prediction results are saved.
- `main.py`: Entry point for the application.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your trained model in `models/` (e.g., `models/my_model.pkl`).

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

The system loads a model, accepts physiological data (Heart Rate, HRV, ECG), runs a prediction, generates an explanation using an LLM, and saves the results to `data/outputs/predictions.csv`.
