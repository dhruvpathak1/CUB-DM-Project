import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Step 1: Load the ETF data from CSV
# Load the data
file_path = '/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/ETF Detailed.csv'
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Convert 'YTD Price Change' to numeric, removing '%' and handling errors
df['YTD Price Change'] = pd.to_numeric(df['YTD Price Change'].str.replace('%', ''), errors='coerce')

# Drop rows with NaN values in 'YTD Price Change'
df.dropna(subset=['YTD Price Change'], inplace=True)

# Step 3: Detect outliers using Z-score
df['Z-Score'] = stats.zscore(df['YTD Price Change'])
outliers = df[(df['Z-Score'] > 3) | (df['Z-Score'] < -3)]

# Step 4: Visualize the results with a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['YTD Price Change'])
plt.title('Box Plot of YTD Price Change with Outliers Highlighted')
plt.xlabel('YTD Price Change (%)')

# Highlight outliers in red
for index, row in outliers.iterrows():
    plt.text(row['YTD Price Change'], 0, 'Outlier', color='red', fontsize=12)

plt.show()
