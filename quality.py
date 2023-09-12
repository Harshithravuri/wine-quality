# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'dataset.csv' with your dataset file)
data = pd.read_csv("dataset.csv")

# Assuming you have columns for features and wine quality, adjust column names accordingly
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Target variable

# Split the data into a training set and a test set (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')

# Optionally, you can also print the coefficients and intercept of the linear regression model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Create a bar plot for feature importance (coefficients)
feature_names = X.columns
coefficients = model.coef_

# Create a DataFrame to store feature names and coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by absolute coefficient values for better visualization
coefficients_df['Absolute Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Absolute Coefficient', ascending=False)

# Create a bar plot for feature importance (coefficients)
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df, palette='viridis')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Coefficients) in Predicting Wine Quality')
plt.show()

# Define a threshold for classifying wine quality
threshold = 6  # You can adjust this threshold based on your criteria

# Classify wines as "Good" or "Not Good" based on the threshold
y_pred_class = np.where(y_pred >= threshold, 'Good', 'Not Good')

# Print the first few rows of the classified predictions
print('Classified Predictions:')
print(y_pred_class[:10])  # Print the first 10 predictions
