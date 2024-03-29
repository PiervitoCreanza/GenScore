import matplotlib.pyplot as plt
import pandas as pd

# Load data from file
file1 = 'retrained_data.csv' # retrained data has yet to be collected 
file2 = 'original_data.csv'  # original data has yet to be collected

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Extract necessary columns for the scatter plot
x_values = data1['x_column'] # will use retrained data
y_values = data2['y_column'] # will use original data

# Create the scatter plot
plt.scatter(x_values, y_values, marker='o', color='g', label='Value of convergence')

# Personalize the graphic
plt.title('Scatter Plot \n GenScore Retrained vs GenScore Original')
plt.xlabel('New Training Values')
plt.ylabel('Original Training Values')
plt.grid(True)
plt.legend()

# Show graphic
plt.show()
