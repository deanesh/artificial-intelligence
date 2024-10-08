import pandas as pd
from sklearn.feature_selection import mutual_info_regression
'''
CATEGORY TYPE: Numeric Data
LIBRARIES USED: Pandas, Scikit-Learn (Feature Selection)
Explanation
Original DataFrame: Shows the movies with their genres, release years, and ratings.
One-Hot Encoded DataFrame: Converts categorical variables into numerical form.
Mutual Information: Displays the mutual information scores for each feature relative to the target variable (Rating). Higher values indicate a stronger relationship between the feature and the target.
Use in Machine Learning
Mutual information can help you select relevant features for your machine learning models, 
leading to better performance and simpler models by removing irrelevant features.
'''

# Sample dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Release_Year': [2010, 2015, 2010, 2020, 2015],
    'Rating': [8.0, 7.5, 9.0, 6.5, 7.0]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# One-hot encode the Genre column
df_encoded = pd.get_dummies(df, columns=['Genre'], drop_first=True)
print("\nOne-Hot Encoded DataFrame:")
print(df_encoded)

# Separate features and target variable
X = df_encoded.drop(columns=['Movie', 'Rating'])
y = df_encoded['Rating']

# Calculate mutual information
mi = mutual_info_regression(X, y)

# Create a DataFrame for better visualization
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
print("\nMutual Information between Features and Target:")
print(mi_df)

'''
Original DataFrame:
      Movie    Genre  Release_Year  Rating
0  Movie A   Action          2010     8.0
1  Movie B   Comedy          2015     7.5
2  Movie C   Action          2010     9.0
3  Movie D    Drama          2020     6.5
4  Movie E   Comedy          2015     7.0

One-Hot Encoded DataFrame:
      Release_Year  Rating  Genre_Comedy  Genre_Drama  Genre_Action
0             2010     8.0              0             0              1
1             2015     7.5              1             0              0
2             2010     9.0              0             0              1
3             2020     6.5              0             1              0
4             2015     7.0              1             0              0

Mutual Information between Features and Target:
            Feature  Mutual Information
0      Release_Year              0.022096
1             Genre_Comedy           0.028099
2             Genre_Drama            0.059106
3             Genre_Action           0.134823

'''
