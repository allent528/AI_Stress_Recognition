# FILE: ai_engine/llm_feedback.py
import os
from .types import PredictionOutput

# Try to import openai, handle if not installed or configured
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def generate_feedback(ml_output: PredictionOutput, history_stats: dict = None) -> str:
    """
    Generates a personalized explanation using an LLM based on ML output and optional history.

    Args:
        ml_output (dict): The output from the ML model (e.g., {'stress_prob': 0.8, ...})
        history_stats (dict, optional): Historical statistics (e.g., {'recent_stress_freq': 0.3}).

    Returns:
        str: The generated feedback text.
    """
    stress_prob = ml_output.get("stress_prob", 0.0)
    
    prompt = (
        f"The model predicted a stress probability of {stress_prob:.2f}. "
    )

    if history_stats:
        recent_freq = history_stats.get('recent_stress_freq', 0.0)
        prompt += f"Historically, the user has been stressed {recent_freq*100:.1f}% of the time recently. "

    prompt += "Explain what this means for the user in simple, actionable terms."

    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not OpenAI or not api_key:
        # Fallback if OpenAI is not available or API key is missing
        return (
            f"[MOCK LLM OUTPUT] Based on a stress probability of {stress_prob:.2f}, "
            "it is recommended to take a short break and practice deep breathing. "
            "(Note: OpenAI API key not found or library missing, using mock response)"
        )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

def generate_historical_feedback(history_summary: dict) -> str:
    """
    Generates feedback based on aggregated historical data.
    
    Args:
        history_summary (dict): Aggregated data (e.g., {'avg_stress_prob': 0.4, 'stress_frequency': 0.2})
        
    Returns:
        str: Personalized feedback based on history.
    """
    avg_stress = history_summary.get("avg_stress_prob", 0.0)
    stress_freq = history_summary.get("stress_frequency", 0.0)
    
    prompt = (
        f"Over the recorded history, the user had an average stress probability of {avg_stress:.2f} "
        f"and was stressed {stress_freq*100:.1f}% of the time. "
        "Provide a summary of their stress trends and 2-3 personalized long-term recommendations."
    )
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not OpenAI or not api_key:
        return (
            f"[MOCK LLM OUTPUT] Historical Analysis: Average stress was {avg_stress:.2f}. "
            "It seems you have periods of high stress. Consider long-term stress management like meditation or regular exercise. "
            "(Note: OpenAI API key not found or library missing, using mock response)"
        )
        
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert health analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating historical feedback: {str(e)}"
