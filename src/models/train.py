import pandas as pd
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier

# --- Path Correction ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End Path Correction ---

from src.data.preprocess import preprocess_data

def train_model():
    """Trains the final model and saves it to disk."""
    # Define paths
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'survey.csv')
    model_save_path = os.path.join(project_root, 'models', 'final_model.pkl')
    columns_save_path = os.path.join(project_root, 'models', 'model_columns.pkl')

    print("Loading raw data...")
    df_raw = pd.read_csv(raw_data_path)

    print("Preprocessing data...")
    df_processed = preprocess_data(df_raw, is_training=True)

    # Separate features and target
    X = df_processed.drop('treatment', axis=1, errors='ignore')
    # The 'treatment' column might not exist if it was dropped during preprocessing
    # Re-map target variable if it exists
    if 'treatment' in df_raw.columns:
        y = df_raw['treatment'].map({'Yes': 1, 'No': 0})
    else: # Fallback in case 'treatment' column name is different or missing
        print("Warning: 'treatment' column not found. Cannot train model.")
        return

    # Store the column list that the model was trained on
    model_columns = X.columns.tolist()
    joblib.dump(model_columns, columns_save_path)
    print(f"Model columns saved to {columns_save_path}")

    # Define the best parameters from our Optuna study
    best_params = {
        'n_estimators': 378,
        'max_depth': 16,
        'min_samples_split': 10,
        'min_samples_leaf': 1
    }

    print("Training final model...")
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X, y)

    # Save the trained model
    joblib.dump(final_model, model_save_path)
    print(f"✅ Final model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()

