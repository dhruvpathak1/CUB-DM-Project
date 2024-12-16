import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Creating and training the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Visualization of the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=features, filled=True)
plt.title('Decision Tree for ETF YTD Price Change Prediction')
plt.show()
