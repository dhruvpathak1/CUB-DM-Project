import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('/Users/dhruvpathak/Desktop/Sem 1/DM/Project/CSV Files/Combined 5 ETF - Sheet1.csv')

# Convert Total Assets to numeric, removing $ and commas
df['Total Assets'] = df['Total Assets'].replace('[\$,]', '', regex=True).astype(float)

# Group by Sector and sum the Total Assets
sector_assets = df.groupby('Sector')['Total Assets'].sum().sort_values(ascending=False)

print(sector_assets)

# Assuming sector_assets is your data
data = sector_assets.sort_values(ascending=False)


# Create a custom color palette
colors = sns.color_palette("pastel", n_colors=len(data))

plt.figure(figsize=(12, 12))
wedges, texts, autotexts = plt.pie(data.values, labels=data.index, colors=colors, autopct=lambda pct: f'{pct:.1f}%', pctdistance=0.80, startangle=120, wedgeprops=dict(width=0.35, edgecolor='white'))

# Add a slight shadow effect
for w in wedges:
    w.set_edgecolor('black')
    w.set_linewidth(3)
    w.set_alpha(0.8)

plt.title('Sectorwise Distribution', fontweight='bold', fontsize=16, loc='center', y=0.50)
plt.axis('equal')

# Customize text properties
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)


plt.tight_layout()
plt.savefig('sector_assets_pie_soothing.png', dpi=300)
plt.close()