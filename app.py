import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the preprocessed dataset
data = pd.read_csv('processed_data.csv')  # Assuming you save the processed data

# Title and description
st.title('Customer Personality Analysis')
st.write("Analyze and visualize customer segments based on their behavior and spending patterns.")

# Sidebar for selecting the number of clusters
k = st.sidebar.slider('Select Number of Clusters', 2, 10, value=4)

# Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(['Cluster'], axis=1))
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]

# Scatter plot of clusters
fig, ax = plt.subplots()
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', ax=ax)
ax.set_title('Customer Segments')
st.pyplot(fig)

# Display summary statistics of each cluster
st.write("Cluster Summary Statistics")
cluster_summary = data.groupby('Cluster').mean()
st.dataframe(cluster_summary)
