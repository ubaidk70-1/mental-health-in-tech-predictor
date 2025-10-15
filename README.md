## Mental Health in Tech: Predictive Model and Web Application
Live Demo: Mental Health Predictor App 🚀

### Project Overview

This project is an end-to-end machine learning solution developed for the PW Skills Mini-Hackathon. The goal is to predict whether an individual working in the tech industry has sought treatment for a mental health condition based on survey data.

The project moves beyond simple prediction to deliver a fully deployed and interpretable model. It provides actionable insights into the factors that influence mental health treatment-seeking behavior, such as workplace support, perceived stigma, and personal history. The final output is an interactive web application where users can input their information and receive a real-time prediction.

### Key Features
+ **In-Depth Data Analysis (EDA):** Comprehensive cleaning, preprocessing, and visualization to uncover key patterns in the data.

+ **Advanced Feature Engineering:** Creation of composite scores like support_score and stigma_score to build a more insightful model.

+ **High-Performance Modeling:** Trained and evaluated multiple models, selecting a tuned Random Forest Classifier that achieved 73% accuracy on unseen test data.

+ **Model Interpretability:** Used SHAP (SHapley Additive exPlanations) to explain the model's decisions, identifying family_history and work_interfere as the most critical predictive features.

+ **Fairness Audit:** Conducted a bias analysis to assess model performance across different demographic subgroups.

+ **Live Web Application:** Deployed the final model as an interactive web application using Flask and Hugging Face Spaces.

Tech Stack
Data Science & Modeling: Python, Pandas, NumPy, Scikit-learn

Advanced Models: XGBoost, LightGBM

Hyperparameter Tuning: Optuna

Model Explainability: SHAP

Web Backend: Flask

Deployment: Hugging Face Spaces, Git

Project Structure
The project is organized into a modular structure to ensure maintainability and reproducibility.

├── data/             # Contains raw and processed datasets
├── models/           # Stores trained model artifacts (.pkl)
├── notebooks/        # Jupyter Notebooks for EDA and experimentation
├── src/              # Source code for the ML pipeline (preprocessing, training, prediction)
├── static/           # CSS and other static assets for the web app
├── templates/        # HTML templates for the Flask application
├── app.py            # Main Flask application file
├── requirements.txt  # Project dependencies
├── report.md         # Detailed project report
└── README.md         # This file

Installation and Usage
To run this project locally, please follow these steps:

Clone the Repository

git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>

Create and Activate a Virtual Environment

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Train the Model (if needed)
The repository includes the pre-trained model. To retrain it, run:

python -m src.models.train

Run the Web Application

python app.py

Open your web browser and navigate to http://127.0.0.1:5000 to use the application.

Model Insights
The final model identified a clear hierarchy of factors influencing the decision to seek mental health treatment. An individual's family history and the degree to which their condition interferes with work are the most dominant predictors. The model's logic is transparent and aligns with real-world understanding, making it a trustworthy tool for generating insights.

For a complete analysis, please see the Project Report.
