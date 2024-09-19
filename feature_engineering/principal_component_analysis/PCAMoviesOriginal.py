import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample movie data (you can load your own dataset here)
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E', 'Movie F', 'Movie G'],
    'IMDB Rating': [8.7, 7.2, 6.5, 9.0, 8.2, 6.8, 7.5],
    'Budget (Millions)': [20, 10, 15, 30, 25, 12, 18],
    'Box Office Revenue (Millions)': [100, 50, 70, 120, 110, 60, 80]
}

df = pd.DataFrame(data)

# Selecting the numerical columns for PCA
X = df[['IMDB Rating', 'Budget (Millions)', 'Box Office Revenue (Millions)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Number of principal components to retain
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance_ratio}")
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
# Create a DataFrame for the principal components
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
loadings = pd.DataFrame( pca.components_.T, columns=component_names, index=X.columns )
print(loadings)
# Add movie names to the PCA transformed data
df_pca['Movie'] = df['Movie']

# Visualize PCA components
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], edgecolors='k', c='blue', s=100)
plt.title('PCA of Movies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i, movie in enumerate(df_pca['Movie']):
    plt.text(df_pca['PC1'][i], df_pca['PC2'][i], movie, fontsize=9, ha='left')

plt.grid(True)
plt.tight_layout()
plt.show()
