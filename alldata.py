import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
datasets_df = pd.read_csv('Fairness Datasets.csv')

# Correct the column name in Fairness Datasets and remove any leading/trailing spaces
datasets_df.rename(columns=lambda x: x.strip(), inplace=True)

# Filter out rows where 'Bias types' or 'Mechanism Type' are empty
filtered_df = datasets_df.dropna(subset=['Bias types', 'Mechanism Type'])

# Split 'Mechanism Type' and 'Bias types' into separate rows when multiple values are separated by commas
filtered_df.loc[:, 'Mechanism Type'] = filtered_df['Mechanism Type'].apply(lambda x: x.split(', '))
filtered_df.loc[:, 'Bias types'] = filtered_df['Bias types'].apply(lambda x: x.split(', '))
exploded_df = filtered_df.explode('Mechanism Type').explode('Bias types')

# Remove duplicates to avoid over-representation of repeated datasets
unique_df = exploded_df.drop_duplicates(subset=['Name', 'Mechanism Type', 'Bias types'])

# Convert categorical columns to category codes and apply minimal jitter
unique_df.loc[:, 'Mechanism Type Code'] = unique_df['Mechanism Type'].astype('category').cat.codes
unique_df.loc[:, 'Bias types Code'] = unique_df['Bias types'].astype('category').cat.codes
unique_df.loc[:, 'Jittered Mechanism Type'] = unique_df['Mechanism Type Code'] + np.random.normal(0, 0.05, size=len(unique_df))
unique_df.loc[:, 'Jittered Bias Types'] = unique_df['Bias types Code'] + np.random.normal(0, 0.05, size=len(unique_df))

# Create the scatter plot with dataset names as hue for color differentiation
plt.figure(figsize=(14, 10))
scatter = sns.scatterplot(data=unique_df, x='Jittered Mechanism Type', y='Jittered Bias Types', hue='Name', style='Name', s=100, alpha=0.7, palette='viridis')

plt.title('Distribution of Bias Types by Mechanism Type')
plt.xlabel('Mechanism Type')
plt.ylabel('Bias Types')
plt.grid(True)
plt.legend(title='Dataset Name', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Adjust margins to accommodate legends and axis labels
plt.subplots_adjust(left=0.12, right=0.825)  # Adjust right margin to prevent the legend from being cut off

# Set tighter y-axis limits based on the categorical codes to reduce gaps
plt.ylim(unique_df['Bias types Code'].min() - 1, unique_df['Bias types Code'].max() + 1)

# Update the x and y-ticks to show the category names instead of codes
plt.xticks(ticks=np.unique(unique_df['Mechanism Type Code']), labels=unique_df['Mechanism Type'].astype('category').cat.categories)
plt.yticks(ticks=np.unique(unique_df['Bias types Code']), labels=unique_df['Bias types'].astype('category').cat.categories)

plt.show()
