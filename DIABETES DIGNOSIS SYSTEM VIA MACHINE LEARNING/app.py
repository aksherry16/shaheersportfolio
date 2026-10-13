from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [
            data['pregnancies'], data['glucose'], data['bloodPressure'],
            data['skinThickness'], data['insulin'], data['bmi'],
            data['dpf'], data['age']
        ]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return jsonify({'prediction': 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
