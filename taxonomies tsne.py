import pandas as pd

# Load the dataset, skip subtitle row
file_path = 'Fairness Taxonomies.csv'
data = pd.read_csv(file_path, header=0, skiprows=[1])  # Adjust if the header and skiprows are different

# Display the first few rows of the dataset
print(data.head())

# Remove the first two columns (not three, based on your last code)
data_cleaned = data.drop(data.columns[:2], axis=1)

# Fill missing values with a placeholder
data_filled = data_cleaned.fillna('Unknown')

# Apply one-hot encoding
data_encoded = pd.get_dummies(data_filled)

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(data_encoded)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.title('t-SNE projection of the Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(tsne_results)

# Plot the clustered t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.title('t-SNE projection with K-means Clustering')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.show()


# Add cluster labels to the data_filled DataFrame
data_filled['Cluster'] = clusters  # Make sure this line executes without errors before moving on

# Now you can safely use the 'Cluster' column
print(data_filled.head())

# Investigate each cluster
for i in range(5):  # Make sure to iterate through the correct number of clusters
    cluster_data = data_filled[data_filled['Cluster'] == i]
    print(f"Cluster {i} Statistics:")
    print(cluster_data.describe(include='all'))
    print("\n")

# Create a plot to visualize the distribution of 'Types of bias' across clusters
plt.figure(figsize=(14, 8))  # Adjust figure size to give more space if needed
ax = sns.countplot(data=data_filled, x='Types of bias', hue='Cluster')
plt.title('Distribution of Types of Bias Across Clusters')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Adjust rotation and alignment

# Improve layout adjustments
plt.tight_layout()

# Optionally, adjust the legend position if it overlaps with any bars
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()