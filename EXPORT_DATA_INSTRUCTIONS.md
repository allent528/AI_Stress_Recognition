# How to Export Your Processed Data

To feed your historical data into the AI engine, you need to export the processed features from your `AI_Proj_phase3.ipynb` notebook.

## Steps

1.  **Open `AI_Proj_phase3.ipynb`**.
2.  **Run all cells** up to the feature extraction part (where `X` and `subject_ids_all` are created).
3.  **Add and Run** the following code in a new cell at the end of the notebook:

```python
import pandas as pd
import os

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Create a DataFrame with User IDs and Features
# 'X' is your feature DataFrame from the notebook
# 'subject_ids_all' is the list of subject IDs created during extraction
export_df = X.copy()
export_df.insert(0, 'userId', subject_ids_all)

# Save to CSV
output_path = "data/processed_user_data.csv"
export_df.to_csv(output_path, index=False)

print(f"Data exported to {output_path}")
print(f"Shape: {export_df.shape}")
```

4.  **Verify**: Check that `data/processed_user_data.csv` exists.

## Next Steps
Once you have this file, I will create a script (`run_batch.py`) to read it, process each user's history, and generate personalized insights.
