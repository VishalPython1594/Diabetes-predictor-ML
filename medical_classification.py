from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the model
model = joblib.load('Random_forest_model_healthcare.pkl')

# Initialize the app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('med.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    form_data = request.form
    age = float(form_data['age'])
    days = float(form_data['days'])
    gender = form_data['gender']
    blood_type = form_data['blood_type']
    medical_condition = form_data['medical_condition']
    admission_type = form_data['admission_type']
    medication = form_data['medication']

    # Map categorical data to one-hot encoding
    input_data = {
        'Age': age,
        'Days Stayed': days,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Blood Type_A-': 1 if blood_type == 'A-' else 0,
        'Blood Type_AB+': 1 if blood_type == 'AB+' else 0,
        'Blood Type_AB-': 1 if blood_type == 'AB-' else 0,
        'Blood Type_B+': 1 if blood_type == 'B+' else 0,
        'Blood Type_B-': 1 if blood_type == 'B-' else 0,
        'Blood Type_O+': 1 if blood_type == 'O+' else 0,
        'Blood Type_O-': 1 if blood_type == 'O-' else 0,
        'Medical Condition_Asthma': 1 if medical_condition == 'Asthma' else 0,
        'Medical Condition_Cancer': 1 if medical_condition == 'Cancer' else 0,
        'Medical Condition_Obesity': 1 if medical_condition == 'Obesity' else 0,
        'Medical Condition_Diabetes': 1 if medical_condition == 'Diabetes' else 0,
        'Medical Condition_Hypertension': 1 if medical_condition == 'Hypertension' else 0,
        'Admission Type_Emergency': 1 if admission_type == 'Emergency' else 0,
        'Admission Type_Urgent': 1 if admission_type == 'Urgent' else 0,
        'Medication_Ibuprofen': 1 if medication == 'Ibuprofen' else 0,
        'Medication_Lipitor': 1 if medication == 'Lipitor' else 0,
        'Medication_Paracetamol': 1 if medication == 'Paracetamol' else 0,
        'Medication_Penicillin': 1 if medication == 'Penicillin' else 0,
    }

    # Create a DataFrame with the expected column order
    expected_order = [
        'Age', 'Days Stayed', 'Gender_Male', 'Blood Type_A-', 'Blood Type_AB+',
        'Blood Type_AB-', 'Blood Type_B+', 'Blood Type_B-', 'Blood Type_O+',
        'Blood Type_O-', 'Medical Condition_Asthma', 'Medical Condition_Cancer',
        'Medical Condition_Diabetes', 'Medical Condition_Hypertension',
        'Medical Condition_Obesity', 'Admission Type_Emergency',
        'Admission Type_Urgent', 'Medication_Ibuprofen', 'Medication_Lipitor',
        'Medication_Paracetamol', 'Medication_Penicillin'
    ]
    input_df = pd.DataFrame([input_data]).reindex(columns=expected_order, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    result_mapping = {0: 'Normal', 1: 'Abnormal', 2: 'Inconclusive'}
    result = result_mapping[prediction[0]]

    # Render result in template
    return render_template('med.html', result=f'The predicted Test result is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
