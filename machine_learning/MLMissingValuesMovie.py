import pandas as pd
import numpy as np

'''
Explanation
Original DataFrame: Displays the movie data with missing values.
Missing Values Count: Shows how many missing values are present in each column.
Handling Missing Values:
The Genre column replaces missing values with 'Unknown'.
The Rating column fills missing values with the mean rating.
The Release Year column fills missing values with the most common year (mode).
The Runtime column fills missing values with the median runtime.
'''

# Sample dataset with missing values
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', np.nan, 'Drama', 'Comedy'],
    'Rating': [8.0, np.nan, 9.0, 6.5, 7.0],
    'Release_Year': [2010, 2015, 2010, np.nan, 2015],
    'Runtime': [120, 95, 110, 130, np.nan]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Fill missing values
df['Genre'].fillna('Unknown', inplace=True)  # Replace missing genres with 'Unknown'
df['Rating'].fillna(df['Rating'].mean(), inplace=True)  # Fill missing ratings with the mean rating
df['Release_Year'].fillna(df['Release_Year'].mode()[0], inplace=True)  # Fill with the most common year
df['Runtime'].fillna(df['Runtime'].median(), inplace=True)  # Fill missing runtime with the median

print("\nDataFrame after handling missing values:")
print(df)

'''
Output
When you run this code, you will see:
The original DataFrame printed with missing values.
A count of missing values in each column.
The DataFrame after handling missing values, showing how the missing entries have been filled.
'''


