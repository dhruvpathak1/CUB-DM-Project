import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/Combined 5 ETF - Sheet1.csv')

# Identify numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

# If you want to include categorical variables, you can use one-hot encoding
# data_encoded = pd.get_dummies(data, columns=categorical_columns)

# For now, let's focus on numeric columns
numeric_data = data[numeric_columns]

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

# If you want to see correlations with P/E ratio and Beta specifically
pe_beta_corr = correlation_matrix[['P/E Ratio', 'Beta']].sort_values(by='P/E Ratio', ascending=False)
print("\nCorrelations with P/E Ratio and Beta:")
print(pe_beta_corr)