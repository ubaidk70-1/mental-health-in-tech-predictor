import pandas as pd
import os
import sys

# --- Path Correction ---
# This ensures that we can use absolute imports from the project root
# Get the absolute path of the directory the current file is in (i.e., .../src/data)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root by going up two levels
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End Path Correction ---

def preprocess_data(data, is_training=False, fitted_columns=None):
    """
    Cleans, encodes, and engineers features for the mental health dataset.
    Can be used for both training and single-instance prediction.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # --- Data Cleaning ---
    # 1. Age Cleaning
    valid_age_median = 31.0 # Pre-calculated from our notebook
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df.loc[(df['Age'] < 18) | (df['Age'] > 75) | (df['Age'].isnull()), 'Age'] = valid_age_median

    # 2. Gender Cleaning
    male_terms = ['Male', 'male', 'M', 'm', 'Make', 'Cis Male', 'Man', 'msle', 'Mail', 'Mal', 'Cis Man', 'Male-ish', 'maile', 'Malr']
    female_terms = ['Female', 'female', 'F', 'f', 'Woman', 'Cis Female', 'Femake', 'woman', 'Female ', 'cis-female/femme', 'femail']
    df['Gender'] = df['Gender'].apply(lambda x: 'Male' if x in male_terms else ('Female' if x in female_terms else 'Other'))

    # 3. Impute missing values with mode (for simplicity in production)
    for col in ['self_employed', 'work_interfere']:
        if col in df.columns and df[col].isnull().any():
            # In a real app, you'd save the mode from the training set
            # For this hackathon, we'll hardcode a common value
            mode_val = 'No' if col == 'self_employed' else 'Sometimes'
            df[col].fillna(mode_val, inplace=True)

    # Drop columns not used in the model
    if 'Timestamp' in df.columns:
        df.drop(['Timestamp', 'comments', 'state'], axis=1, inplace=True, errors='ignore')


    # --- Feature Engineering & Encoding ---
    # 1. Manual Encoding
    binary_map = {'Yes': 1, 'No': 0, "Don't know": 0}
    for col in ['benefits', 'wellness_program', 'seek_help', 'anonymity']:
        df[col] = df[col].map(binary_map)

    leave_map = {'Very easy': 4, 'Somewhat easy': 3, "Don't know": 2, 'Somewhat difficult': 1, 'Very difficult': 0}
    df['leave'] = df['leave'].map(leave_map)

    stigma_map = {'Yes': 2, 'Maybe': 1, 'No': 0}
    for col in ['mental_health_consequence', 'phys_health_consequence']:
        df[col] = df[col].map(stigma_map)

    interfere_map = {'Often': 3, 'Sometimes': 2, 'Rarely': 1, 'Never': 0}
    df['work_interfere'] = df['work_interfere'].map(interfere_map)

    employee_map = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
    df['no_employees'] = df['no_employees'].map(employee_map)

    # 2. Engineered Features
    support_columns = ['benefits', 'wellness_program', 'seek_help', 'anonymity', 'leave']
    df['support_score'] = df[support_columns].sum(axis=1)

    stigma_columns = ['mental_health_consequence', 'phys_health_consequence']
    df['stigma_score'] = df[stigma_columns].sum(axis=1)

    bins = [18, 30, 40, 50, 75]
    labels = ['18-30', '31-40', '41-50', '51-75']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    if 'Age' in df.columns:
        df.drop('Age', axis=1, inplace=True)

    # Remaining simple text columns to encode
    for col in ['self_employed', 'family_history', 'remote_work', 'tech_company', 'care_options', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']:
        df[col] = pd.Categorical(df[col]).codes


    # --- One-Hot Encoding ---
    cols_to_ohe = ['Country', 'Gender', 'age_group']
    df = pd.get_dummies(df, columns=cols_to_ohe, drop_first=True)


    # --- Align Columns for Prediction ---
    if not is_training and fitted_columns is not None:
        # For prediction, ensure the columns match the training set exactly
        missing_cols = set(fitted_columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        # Ensure the order is the same
        df = df[fitted_columns]

    return df

