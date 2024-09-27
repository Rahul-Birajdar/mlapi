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

# Load the dataset
data = pd.read_csv('CP.csv')

# Get unique states (as original strings)
fertilizer = data['Fertilizer'].unique()
print("Fertilizer:", fertilizer)


######################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset (ensure the Fertilizer column is present)
data = pd.read_csv('CP.csv')

# Display dataset structure (optional)
print(data.head())

# Split the data into features (X) and target (y)
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
naive_bayes = GaussianNB()

# Train the models
logreg.fit(X_train_scaled, y_train)
dtree.fit(X_train_scaled, y_train)
rforest.fit(X_train_scaled, y_train)
naive_bayes.fit(X_train_scaled, y_train)

# Make predictions using the test data
logreg_pred = logreg.predict(X_test_scaled)
dtree_pred = dtree.predict(X_test_scaled)
rforest_pred = rforest.predict(X_test_scaled)
nb_pred = naive_bayes.predict(X_test_scaled)

# Calculate accuracy for each model
logreg_acc = accuracy_score(y_test, logreg_pred)
dtree_acc = accuracy_score(y_test, dtree_pred)
rforest_acc = accuracy_score(y_test, rforest_pred)
nb_acc = accuracy_score(y_test, nb_pred)

# Print accuracies
print(f'Logistic Regression Accuracy: {logreg_acc:.2f}')
print(f'Decision Tree Accuracy: {dtree_acc:.2f}')
print(f'Random Forest Accuracy: {rforest_acc:.2f}')
print(f'Naive Bayes Accuracy: {nb_acc:.2f}')

# Use Naive Bayes for final prediction and fertilizer recommendation
X_test['Predicted Crop'] = nb_pred  # Naive Bayes predictions

# Map the predicted crop to the corresponding fertilizer from the original dataset
# Ensure the Crop and Fertilizer columns have unique combinations
# We group by Crop to handle any duplicates and keep the first fertilizer for each crop
fertilizer_map = data.groupby('Crop')['Fertilizer'].first()

# Map the predicted crop to fertilizer
X_test['Recommended Fertilizer'] = X_test['Predicted Crop'].map(fertilizer_map)

# Display the test data along with predicted crop and recommended fertilizer
print(X_test[['Predicted Crop', 'Recommended Fertilizer']])

# Save the scaler and Naive Bayes model as pickle files
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(naive_bayes, f)

print("Scaler and model saved successfully.")
