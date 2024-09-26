import pandas as pd

# Load the dataset
data = pd.read_csv('CP.csv')

# Get unique states (as original strings)
unique_states = data['STATE'].unique()
print("Unique States:", unique_states)

# Get unique crops (as original strings)
unique_crops = data['Crop'].unique()
print("Unique Crops:", unique_crops)

# Find highest and lowest values for each parameter
parameters = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']

for param in parameters:
    highest = data[param].max()
    lowest = data[param].min()
    print(f"Highest {param}: {highest}")
    print(f"Lowest {param}: {lowest}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('CP.csv')

# Ensure that only the relevant columns are used (excluding 'STATE')
data = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall', 'Crop']]

# Split the data into features and target
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']]
y = data['Crop']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
logreg = LogisticRegression(max_iter=1000)
dtree = DecisionTreeClassifier()
rforest = RandomForestClassifier()

# Train the models
logreg.fit(X_train_scaled, y_train)
dtree.fit(X_train_scaled, y_train)
rforest.fit(X_train_scaled, y_train)

# Make predictions using scaled test data
logreg_pred = logreg.predict(X_test_scaled)
dtree_pred = dtree.predict(X_test_scaled)
rforest_pred = rforest.predict(X_test_scaled)

# Calculate accuracy for each model
logreg_acc = accuracy_score(y_test, logreg_pred)
dtree_acc = accuracy_score(y_test, dtree_pred)
rforest_acc = accuracy_score(y_test, rforest_pred)

# Print accuracies
print(f'Logistic Regression Accuracy: {logreg_acc:.2f}')
print(f'Decision Tree Accuracy: {dtree_acc:.2f}')
print(f'Random Forest Accuracy: {rforest_acc:.2f}')

# Save the scaler and Random Forest model as pickle files
with open('s.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('m.pkl', 'wb') as f:
    pickle.dump(rforest, f)

print("Scaler and Random Forest model saved successfully.")

