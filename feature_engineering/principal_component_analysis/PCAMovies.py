import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
CATEGORY TYPE: Numeric Data
LIBRARIES USED: Pandas, Scikit-learn, matplotlib
Explanation
Original DataFrame: Displays the movie data with features.
One-Hot Encoding: Converts the Genre categorical variable into numerical format.
Standardization: Scales the numerical features to have mean 0 and variance 1.
PCA Results: Reduces the dataset to two principal components that capture the most variance.
Visualization: The scatter plot helps visualize how the movies relate to each other based on the selected features.
'''

# Sample dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'], 'Rating': [8.0, 7.5, 9.0, 6.5, 7.0],
    'Release_Year': [2010, 2015, 2010, 2020, 2015], 'Runtime': [120, 95, 110, 130, 85]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# One-hot encode Genre column (drop_first=True will delete the first column to create multi collinearity)
df_encoded = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# Separate features
X = df_encoded.drop(columns=['Movie'])
print(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nScaled Features:")
print(X_scaled)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Results:")
print(X_pca)

# Create a DataFrame for PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Movie'] = df['Movie']
print(df_pca)

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], marker='o')

for i, txt in enumerate(df_pca['Movie']):
    plt.annotate(txt, (df_pca['PC1'][i], df_pca['PC2'][i]))

plt.title('PCA of Movie Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

'''
Output
When you run this code, you will see:
The original DataFrame printed.
Scaled features.
PCA results in two principal components.
A scatter plot showing the movies in the reduced PCA space.
'''
