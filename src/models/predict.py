import pandas as pd
import joblib
import os
import sys

# --- Path Correction ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End Path Correction ---

from src.data.preprocess import preprocess_data

# Define paths to the saved model and columns
MODEL_PATH = os.path.join(project_root, 'models', 'final_model.pkl')
COLUMNS_PATH = os.path.join(project_root, 'models', 'model_columns.pkl')

# Load the model and columns once when the script is loaded
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model or columns file not found. Please run train.py first.")
    model = None
    model_columns = None


def make_prediction(input_data):
    """
    Makes a prediction using the saved model.
    Input_data should be a dictionary.
    """
    if model is None or model_columns is None:
        return {"error": "Model not loaded. Please check server logs."}

    # Preprocess the input data
    processed_input = preprocess_data(input_data, is_training=False, fitted_columns=model_columns)

    # Get prediction probability for the 'Yes' class
    prediction_proba = model.predict_proba(processed_input)[:, 1]
    confidence = f"{prediction_proba[0]:.2%}"

    # Get the final prediction
    prediction = 'Yes' if prediction_proba[0] > 0.5 else 'No'

    return {
        "prediction": prediction,
        "confidence_probability": confidence
    }

# Example usage for testing the script directly
if __name__ == '__main__':
    sample_input = {
        'Age': 35, 'Gender': 'Female', 'Country': 'United States',
        'self_employed': 'No', 'family_history': 'Yes', 'work_interfere': 'Sometimes',
        'no_employees': '26-100', 'remote_work': 'No', 'tech_company': 'Yes',
        'benefits': 'Yes', 'care_options': 'Not sure', 'wellness_program': 'No',
        'seek_help': "Don't know", 'anonymity': 'Yes', 'leave': 'Somewhat easy',
        'mental_health_consequence': 'No', 'phys_health_consequence': 'No',
        'coworkers': 'Some of them', 'supervisor': 'Yes', 'mental_health_interview': 'No',
        'phys_health_interview': 'Maybe', 'mental_vs_physical': 'Yes',
        'obs_consequence': 'No'
    }
    result = make_prediction(sample_input)
    print("Prediction Result:", result)

