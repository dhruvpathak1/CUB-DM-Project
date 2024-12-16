import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/Combined 5 ETF - Sheet1.csv')

# Print column names to see what's available
print("Available columns:", df.columns.tolist())

# Set target and features
target = 'Beta'
features = ['YTD Price Change', 'P/E Ratio']

# Prepare the data
X = df[features]
y = df[target]

# Handle non-numeric data and remove rows with NaN values
X = X.replace('[\$,%]', '', regex=True).astype(float)
y = pd.to_numeric(y, errors='coerce')
data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data[target]

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
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Feature importance
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature: {features[i]}, Score: {v:.5f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Beta")
plt.ylabel("Predicted Beta")
plt.title("Actual vs Predicted Beta")
plt.tight_layout()
plt.show()
