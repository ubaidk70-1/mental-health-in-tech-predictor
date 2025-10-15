# Mental Health in Tech: Predictive Model and Web Application

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://ubaidkha-mental-health-in-it.hf.space) 


---

## Project Overview

This project is an **end-to-end machine learning solution** developed for the **PW Skills Mini-Hackathon**.  
The goal is to predict whether an individual working in the tech industry has **sought treatment for a mental health condition** based on survey data.

The project goes beyond simple prediction — it delivers a **fully deployed and interpretable model** that offers **actionable insights** into the factors influencing mental health treatment-seeking behavior, such as workplace support, perceived stigma, and personal history.

The final deliverable is an **interactive web application** where users can input their details and receive **real-time predictions**.

---

## Key Features

- ** In-Depth Data Analysis (EDA):**  
  Comprehensive data cleaning, preprocessing, and visualization to uncover key behavioral patterns.

- ** Advanced Feature Engineering:**  
  Created composite metrics like `support_score` and `stigma_score` to capture nuanced psychological and workplace dynamics.

- ** High-Performance Modeling:**  
  Evaluated multiple ML models and selected a **tuned Random Forest Classifier**, achieving **73% accuracy** on unseen test data.

- ** Model Interpretability:**  
  Leveraged **SHAP (SHapley Additive exPlanations)** to explain model decisions — identifying `family_history` and `work_interfere` as the most influential features.

- ** Fairness Audit:**  
  Conducted bias analysis to assess model performance across demographic subgroups for ethical AI considerations.

- ** Live Web Application:**  
  Deployed the final model as an **interactive Flask web app** on **Hugging Face Spaces**.

---

##  Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Data Science & Modeling** | Python, Pandas, NumPy, Scikit-learn |
| **Advanced Models** | XGBoost, LightGBM |
| **Hyperparameter Tuning** | Optuna |
| **Model Explainability** | SHAP |
| **Web Backend** | Flask |
| **Deployment** | Hugging Face Spaces, Git |

---

##  Project Structure

```
├── data/ # Contains raw and processed datasets
├── models/ # Stores trained model artifacts (.pkl)
├── notebooks/ # Jupyter Notebooks for EDA and experimentation
├── src/ # Source code for the ML pipeline (preprocessing, training, prediction)
├── static/ # CSS and other static assets for the web app
├── templates/ # HTML templates for the Flask application
├── app.py # Main Flask application file
├── requirements.txt # Project dependencies
├── report.md # Detailed project report
└── README.md # This file

```


---

##  Installation and Usage

### 1️ Clone the Repository
```
git clone https://github.com/ubaidk70-1/mental-health-in-tech-predictor.git
cd mental-health-in-tech-predictor
```

### 2️ Create and Activate a Virtual Environment
```
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

```

### 3️ Install Dependencies
```
pip install -r requirements.txt
```

### 4️ Train the Model
A pre-trained model is already included, but you can retrain it using:
```
python -m src.models.train
```
### 5️ Run the Web Application
```
python app.py
```
Then, open your browser and navigate to:

└── http://127.0.0.1:5000


## Model Insights

The final model highlights a clear hierarchy of factors influencing whether an individual seeks mental health treatment.
`family_history` and `work_interfere` emerged as the strongest predictors, aligning closely with real-world observations.

This transparency, backed by SHAP explainability, ensures that the model is both trustworthy and interpretable, making it a valuable decision-support tool for organizations aiming to improve workplace mental health initiatives.

## Report

For an in-depth technical explanation, model evaluation, and visualizations, refer to:

└──  report.pdf

### Author

Md Ubaid Khan

*Data Analyst & Machine Learning Enthusiast*
