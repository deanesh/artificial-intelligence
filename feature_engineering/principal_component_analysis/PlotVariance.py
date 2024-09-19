import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_variance(pca):
    # Calculate variance explained by each principal component
    variance = pca.explained_variance_ratio_
    # Cumulative variance explained
    cum_variance = np.cumsum(variance)

    # Plotting the variance explained by each component
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(variance) + 1), variance, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(variance) + 1), cum_variance, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance Ratio per Principal Component')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Sample movie data with factors influencing budget
data = {
    'Movie': ['A', 'B', 'C', 'D', 'E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'Production_Size': ['Large', 'Small', 'Medium', 'Medium', 'Large'],
    'Marketing_Budget': [10, 5, 8, 12, 9],
    'Cast_Expenses': [20, 15, 18, 22, 17],
    'Director_Salary': [5, 3, 4, 6, 4],
    'Budget (Millions)': [100, 80, 60, 110, 95]
}
# Create a DataFrame
df = pd.DataFrame(data)
# Selecting numeric columns for PCA
numeric_cols = ['Director_Salary', 'Marketing_Budget', 'Cast_Expenses']
X = df[numeric_cols].values
# Example usage
# Assuming pca0 is already fitted with PCA on your data
pca = PCA(n_components=3)  # Specify the number of components as needed
pca.fit(X)  # Fit PCA on your data X

# Plotting variance explained
plot_variance(pca)
