import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Load the dataset
data = pd.read_csv('Fairness Datasets.csv')

# Function to expand both comma-separated and semicolon-separated entries into separate rows
def expand_entries(df, column_name):
    return (
        df.drop(column_name, axis=1)
          .join(
              df[column_name]
                .str.replace(';', ',')  # Replace semicolons with commas to unify the split
                .str.split(', ')
                .explode()
                .dropna()
                .reset_index(drop=True)
          )
    )

# Expand 'Bias Cases/Protected Attributes' entries
data_expanded = expand_entries(data, 'Bias Cases/Protected Attributes')

# Group by `Name` and concatenate unique `Area/Domain` and `Bias Cases/Protected Attributes`
data_grouped = data_expanded.groupby('Name').agg({
    'Area/Domain': lambda x: '; '.join(sorted(set(x.dropna()))),
    'Bias Cases/Protected Attributes': lambda x: '; '.join(sorted(set(x.dropna())))
}).reset_index()

# Filter out rows where either 'Area/Domain' or 'Bias Cases/Protected Attributes' are empty
data_filtered = data_grouped[(data_grouped['Area/Domain'] != '') & (data_grouped['Bias Cases/Protected Attributes'] != '')]

# Reset index to ensure proper indexing
data_filtered.reset_index(drop=True, inplace=True)

# Further explode 'Bias Cases/Protected Attributes' after grouping and resetting index
data_filtered_expanded = expand_entries(data_filtered, 'Bias Cases/Protected Attributes')

# Drop duplicates if any after explosion
data_filtered_expanded.drop_duplicates(inplace=True)

# Re-prepare the plot data
x = data_filtered_expanded['Bias Cases/Protected Attributes']
y = data_filtered_expanded['Area/Domain']
labels = data_filtered_expanded['Name']

# Create the plot
plt.figure(figsize=(16, 10))
texts = []
for i, label in enumerate(labels):
    plt.scatter(x.iloc[i], y.iloc[i], marker='o')
    texts.append(plt.text(x.iloc[i], y.iloc[i], f' {label}', fontsize=9, va='center'))

plt.title('Datasets by Protected Attribute and Area/Domain')
plt.xlabel('Protected Attribute')
plt.ylabel('Area/Domain')
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.grid(True)

# Adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()  # Adjust layout to make room for label rotations
plt.show()



# Count Plot of Fairness Metrics
# Function to expand both comma-separated and semicolon-separated entries into separate rows
def expand_entries(df, column_name):
    return (
        df.drop(column_name, axis=1)
          .join(
              df[column_name]
                .str.replace(';', ',')  # Replace semicolons with commas to unify the split
                .str.split(', ')
                .explode()
                .dropna()
                .reset_index(drop=True)
          )
    )

# Expand 'Fairness metrics' entries
data_fairness_expanded = expand_entries(data, 'Fairness metrics')

# Create a count plot of Fairness Metrics
fairness_metrics_counts = data_fairness_expanded['Fairness metrics'].value_counts()

# Plotting the count plot for Fairness Metrics
plt.figure(figsize=(12, 6))
fairness_metrics_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency of Fairness Metrics Across Datasets')
plt.xlabel('Fairness Metrics')
plt.ylabel('Frequency')
# Adjust the rotation and alignment of x-axis labels
plt.xticks(rotation=45, ha='right')  # horizontal alignment set to 'right'
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

