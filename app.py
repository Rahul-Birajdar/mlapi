from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the dataset and preprocess it.\venv\Scripts\activate

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


@app.route('/')
def home():
    states = label_encoder.classes_
    return render_template('index.html', states=states)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        STATE = request.form['STATE']

        if STATE not in state_mapping:
            return jsonify({'error': 'Invalid state provided'}), 400

        state_encoded = state_mapping[STATE]

        input_data = np.array([[Temperature, Humidity, state_encoded]])
        input_data = scaler.transform(input_data)

        rforest_pred = rforest.predict(input_data)[0]

        return jsonify({'Random Forest': rforest_pred})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
