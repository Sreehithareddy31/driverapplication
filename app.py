from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model_path = 'best_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define categorical status mapping
status_mapping = {
    0: 'Incomplete',
    1: 'Approved - License Issued',
    2: 'Denied',
    3: 'Under Review',
    4: 'Pending Fitness Interview'
}

# Define mappings for categorical features
feature_mappings = {
    'type': {'HDR': 1, 'not Needed': 0},
    'drug_test': {'Complete': 1, 'Needed': 0},
    'fru_interview': {'Applicable': 1, 'Not Appilcable': 0},
    'wav_course': {'Complete': 1, 'Needed': 0},
    'defensive_driving': {'Complete': 1, 'Needed': 0},
    'driver_exam': {'Complete': 1, 'Needed': 0},
    'medical_clearance': {'Complete': 1, 'Needed': 0},
    'other_requests': {'Finger prints': 1, 'Not Needed': 0},
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and process inputs
        type = request.form.get('type')
        App_Date = request.form.get('App_Date') 
        drug_test = request.form.get('drug_test')
        fru_interview = request.form.get('fru_interview')
        wav_course = request.form.get('wav_course')
        defensive_driving = request.form.get('defensive_driving')
        driver_exam = request.form.get('driver_exam')
        medical_clearance = request.form.get('medical_clearance')
        other_requests = request.form.get('other_requests')
        last_updated = request.form.get('last_updated')
        app_date_str = request.form.get('last_updated')

        numerical_features = [
            feature_mappings['type'].get(type, -1),
            feature_mappings['drug_test'].get(drug_test, -1),
            feature_mappings['fru_interview'].get(fru_interview, -1),
            feature_mappings['wav_course'].get(wav_course, -1),
            feature_mappings['defensive_driving'].get(defensive_driving, -1),
            feature_mappings['driver_exam'].get(driver_exam, -1),
            feature_mappings['medical_clearance'].get(medical_clearance, -1),
            feature_mappings['other_requests'].get(other_requests, -1),
        ]

        reference_date = datetime(2000, 1, 1)
        app_date = datetime.strptime(app_date_str, '%Y-%m-%d')
        days_since_reference = (app_date - reference_date).days
        numerical_features.append(days_since_reference)
        numerical_features = np.array(numerical_features).reshape(1, -1)
        
        prediction = model.predict(numerical_features)
        prediction_status = status_mapping.get(int(prediction[0]), 'Unknown Status')
        return render_template('result.html', prediction=prediction_status)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

