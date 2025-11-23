# FILE: ai_engine/types.py
from typing import Any, Dict, Union

# Lightweight type aliases
Model = Any
PredictionInput = Dict[str, Union[float, int]]
PredictionOutput = Dict[str, Union[float, bool]]
