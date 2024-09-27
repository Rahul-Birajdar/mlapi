from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the dataset and preprocess it
data = pd.read_csv('CP.csv')

# Load the scaler and trained models
with open('S.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('M.pkl', 'rb') as f:
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
        required_fields = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {", ".join([field for field in required_fields if field not in data])}'}), 400

        # Get the input values from the JSON
        Nitrogen = float(data['Nitrogen'])
        Phosphorus = float(data['Phosphorus'])
        Potassium = float(data['Potassium'])
        Temperature = float(data['Temperature'])
        Humidity = float(data['Humidity'])
        Rainfall = float(data['Rainfall'])

        # Prepare the input for the model
        input_data = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Rainfall]])
        input_data = scaler.transform(input_data)

        # Make the prediction using the Random Forest model
        nb_pred = naive_bayes.predict(input_data)[0]

        # Return the predicted crop
        return jsonify({'Crop': f'Crop - {nb_pred}'})
    
    except KeyError as ke:
        return jsonify({'error': f'Missing key: {ke}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
