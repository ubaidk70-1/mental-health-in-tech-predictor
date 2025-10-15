from flask import Flask, request, render_template
import os
import sys

# --- Path Correction ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End Path Correction ---

from src.models.predict import make_prediction

# Explicitly tell Flask where to find the templates and static folders
app = Flask(__name__,
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = None
    try:
        form_data = request.form.to_dict()
        form_data['Age'] = int(form_data['Age'])
        prediction_result = make_prediction(form_data)
    except Exception as e:
        prediction_result = {'error': f"An error occurred: {e}"}
    return render_template('result.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

