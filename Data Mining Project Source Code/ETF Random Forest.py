import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/ETF Detailed.csv'
data = pd.read_csv(file_path)

# Data Cleaning
data['Total Assets'] = data['Total Assets'].replace({'\$': '', ',': ''}, regex=True).astype(float)
data['YTD Price Change'] = data['YTD Price Change'].str.replace('%', '').astype(float)
data['P/E Ratio'] = pd.to_numeric(data['P/E Ratio'], errors='coerce')
data['Beta'] = pd.to_numeric(data['Beta'], errors='coerce')
data['RSI'] = pd.to_numeric(data['RSI'], errors='coerce')

# Selecting features and target variable
features = ['RSI', 'P/E Ratio', 'Beta', 'Total Assets']
X = data[features].fillna(0)  # Fill missing values with 0 for simplicity
y = data['YTD Price Change']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Print accuracy metrics in terminal
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', label='Perfect Prediction')  # Diagonal line for reference
plt.xlabel('Actual YTD Price Change')
plt.ylabel('Predicted YTD Price Change')
plt.title('Actual vs Predicted YTD Price Change')
plt.legend()
plt.grid()
plt.show()
