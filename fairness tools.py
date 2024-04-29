import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Fairness Tools.csv'
fairness_tools_df = pd.read_csv(file_path)

# Display the first few rows of the dataset and its column names
fairness_tools_df.head(), fairness_tools_df.columns

# Clean up any trailing spaces in column names
fairness_tools_df.columns = fairness_tools_df.columns.str.strip()

# Set a modern theme with grid lines
sns.set_theme(style="whitegrid")

# Function to split comma-separated entries into separate rows
def expand_comma_separated_values(df, column_name):
    return (
        df.dropna(subset=[column_name])
          .set_index('Name')[column_name]
          .str.split(', ', expand=True)
          .stack()
          .reset_index()
          .drop(columns='level_1')
          .rename(columns={0: column_name})
          .drop_duplicates()
    )

# Apply the function to the 'Data types' column
expanded_data_types = expand_comma_separated_values(fairness_tools_df[['Name', 'Data types']], 'Data types')

# Count the frequency of each data type used
data_types_counts = expanded_data_types['Data types'].value_counts()

# Calculate the percentage of each data type
data_types_percentage = data_types_counts / data_types_counts.sum() * 100

# Plotting the percentage of tools by expanded data types used
plt.figure(figsize=(10, 8))
sns.barplot(x=data_types_percentage.values, y=data_types_percentage.index, palette='viridis')
plt.title('Percentage of Tools by Data Types Used')
plt.xlabel('Percentage of Tools (%)')
plt.ylabel('Data Types')
plt.show()

# Apply the function to the 'Metric types' and 'ML task' columns
expanded_metric_types = expand_comma_separated_values(fairness_tools_df[['Name', 'Metric types']], 'Metric types')
expanded_ml_tasks = expand_comma_separated_values(fairness_tools_df[['Name', 'ML task']], 'ML task')

# Plotting the count of tools by expanded metric types used
plt.figure(figsize=(10, 8))
sns.countplot(y='Metric types', data=expanded_metric_types, palette='cubehelix', order=expanded_metric_types['Metric types'].value_counts().index)
plt.title('Count of Tools by Metric Types Used')
plt.xlabel('Count of Tools')
plt.ylabel('Metric Types')
plt.show()

# Plotting the count of tools by expanded ML tasks addressed
plt.figure(figsize=(15, 8))
sns.countplot(y='ML task', data=expanded_ml_tasks, palette='coolwarm', order=expanded_ml_tasks['ML task'].value_counts().index)
plt.title('Count of Tools by ML Tasks Addressed')
plt.xlabel('Count of Tools')
plt.ylabel('ML Tasks')
plt.show()


# Data Type(s) used by each tool
# Filter out tools with specified data types only
specified_data_types = expanded_data_types[expanded_data_types['Data types'].notna()]

# Plotting each tool with the data types it uses using a strip plot
plt.figure(figsize=(18, 10))
sns.stripplot(y='Name', x='Data types', data=specified_data_types, palette='Paired', dodge=False, jitter=False, size=10)
plt.title('Data Types Used by Each Tool')
plt.xlabel('Data Types')
plt.ylabel('Tool Name')
plt.show()


