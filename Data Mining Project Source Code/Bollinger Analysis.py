import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from pandas.plotting import table

# Load the dataset
data = pd.read_csv('/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/Combined 5 ETF - Sheet1.csv')

# Convert relevant columns to numeric after cleaning dollar signs and commas
data['Previous Closing Price'] = data['Previous Closing Price'].str.replace('$', '').str.replace(',', '').astype(float)
data['Lower Bollinger'] = data['Lower Bollinger'].str.replace('$', '').str.replace(',', '').astype(float)
data['Upper Bollinger'] = data['Upper Bollinger'].str.replace('$', '').str.replace(',', '').astype(float)

# Define a function to categorize the price
def categorize_price(row):
    if row['Previous Closing Price'] < row['Lower Bollinger']:
        return 'Below Bollinger Range'
    elif row['Previous Closing Price'] > row['Upper Bollinger']:
        return 'Above Bollinger Range'
    else:
        return 'Within Bollinger Range'

# Apply the function to create a new column
data['Bollinger Category'] = data.apply(categorize_price, axis=1)

# Group by Sector and ETF Company to tabulate the results
sector_tabulation = data.groupby(['Sector', 'Bollinger Category']).size().unstack(fill_value=0)
etf_tabulation = data.groupby(['ETF Company', 'Bollinger Category']).size().unstack(fill_value=0)

# Print the tabulations in a fancy format
print("Sector Tabulation:\n")
print(tabulate(sector_tabulation, headers='keys', tablefmt='grid'))

print("\nETF Company Tabulation:\n")
print(tabulate(etf_tabulation, headers='keys', tablefmt='grid'))


