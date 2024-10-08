import pandas as pd

'''
CATEGORY TYPE: Numeric Data
LIBRARIES USED: Pandas
Explanation
Original DataFrame: Displays the movies with their genres, release years, and ratings.
New Features:
Age: Calculates how long ago each movie was released.
Is_Popular_Genre: Indicates if a genre is popular based on its average rating (1 for popular, 0 for not).
Use in Machine Learning
These newly created features (Age and Is_Popular_Genre) can be used as input variables in machine learning models, 
potentially improving predictive performance by providing additional relevant information.
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

# Feature Engineering
current_year = 2024

# Create Age feature
df['Age'] = current_year - df['Release_Year']

# Calculate the average rating for each genre
genre_popularity = df.groupby('Genre')['Rating'].mean()

# Create Is_Popular_Genre feature (assuming a genre is popular if its average rating is above 7.5)
df['Is_Popular_Genre'] = df['Genre'].map(lambda x: 1 if genre_popularity[x] > 7.5 else 0)

print("\nDataFrame with New Features:")
print(df)

'''
Original DataFrame:
      Movie    Genre  Release_Year  Rating
0  Movie A   Action          2010     8.0
1  Movie B   Comedy          2015     7.5
2  Movie C   Action          2010     9.0
3  Movie D    Drama          2020     6.5
4  Movie E   Comedy          2015     7.0

DataFrame with New Features:
      Movie    Genre  Release_Year  Rating  Age  Is_Popular_Genre
0  Movie A   Action          2010     8.0   14                  1
1  Movie B   Comedy          2015     7.5    9                  0
2  Movie C   Action          2010     9.0   14                  1
3  Movie D    Drama          2020     6.5    4                  0
4  Movie E   Comedy          2015     7.0    9                  0

'''
