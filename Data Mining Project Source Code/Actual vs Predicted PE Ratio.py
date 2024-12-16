import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/Combined 5 ETF - Sheet1.csv')

# Print column names to see what's available
print("Available columns:", df.columns.tolist())

# Assume we're predicting 'P/E Ratio' based on other features
target = 'P/E Ratio'
features = ['Beta', 'Total Assets']  # Adjust these based on your available columns

# Prepare the data
X = df[features]
y = df[target]

# Handle non-numeric data (e.g., currency strings)
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].str.replace(r'[\$,]', '', regex=True).astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature: {features[i]}, Score: {v:.5f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual P/E Ratio")
plt.ylabel("Predicted P/E Ratio")
plt.title("Actual vs Predicted P/E Ratio")
plt.tight_layout()
plt.show()
