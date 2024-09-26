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

    
