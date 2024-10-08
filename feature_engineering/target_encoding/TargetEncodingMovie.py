import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
'''
Explanation
Original DataFrame: Shows the movies, genres, and ratings.
Training and Testing Data: The dataset is split into training and testing sets.
Encoded Training Data: The "Genre" column is replaced with the mean ratings for each genre in the training data.
Encoded Testing Data: The testing data has been transformed based on the encoding learned from the training data.
This method prepares your data for machine learning by effectively handling categorical features and ensuring 
that the model can generalize well on unseen data.
'''

# Sample dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Rating': [8.0, 7.5, 9.0, 6.5, 7.0]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Split into features and target
X = df[['Genre']]
y = df['Rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data:")
print(X_train)
print("\nTesting Data:")
print(X_test)

# Create a target encoder
encoder = ce.TargetEncoder(cols=['Genre'])

# Fit and transform the training data
X_train_encoded = encoder.fit_transform(X_train, y_train)

# Transform the testing data
X_test_encoded = encoder.transform(X_test)

print("\nEncoded Training Data:")
print(X_train_encoded)
print("\nEncoded Testing Data:")
print(X_test_encoded)

'''
Original DataFrame:
      Movie    Genre  Rating
0  Movie A   Action     8.0
1  Movie B   Comedy     7.5
2  Movie C   Action     9.0
3  Movie D    Drama     6.5
4  Movie E   Comedy     7.0

Training Data:
      Genre
0   Action
2   Action
1   Comedy
4   Comedy

Testing Data:
      Genre
3    Drama

Encoded Training Data:
      Genre
0   8.5
2   8.5
1   7.25
4   7.25

Encoded Testing Data:
      Genre
3    6.5

'''
