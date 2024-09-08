from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the dataset and preprocess it
data = pd.read_csv('CP.csv')
label_encoder = LabelEncoder()
data['STATE'] = label_encoder.fit_transform(data['STATE'])

# Create a dictionary to reverse map the label encoding
state_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Load the scaler and trained models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('rforest_model.pkl', 'rb') as f:
    rforest = pickle.load(f)

# Add a route for the home page
@app.route('/')
def home():
    return "API is running. Use POST /predict to get crop predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the request
        data = request.get_json()

        # Validate input data
        required_fields = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall', 'STATE']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {", ".join([field for field in required_fields if field not in data])}'}), 400

        # Get the input values from the JSON
        Nitrogen = float(data['Nitrogen'])
        Phosphorus = float(data['Phosphorus'])
        Potassium = float(data['Potassium'])
        Temperature = float(data['Temperature'])
        Humidity = float(data['Humidity'])
        Rainfall = float(data['Rainfall'])
        STATE = data['STATE']

        # Validate the state
        if STATE not in state_mapping:
            return jsonify({'error': 'Invalid state provided'}), 400

        # Encode the state using label encoding
        state_encoded = state_mapping[STATE]

        # Prepare the input for the model
        input_data = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Rainfall, state_encoded]])
        input_data = scaler.transform(input_data)

        # Make the prediction using the Random Forest model
        rforest_pred = rforest.predict(input_data)[0]

        # Return the predicted crop
        return jsonify({'Crop': f'Crop - {rforest_pred}'})
    
    except KeyError as ke:
        return jsonify({'error': f'Missing key: {ke}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
