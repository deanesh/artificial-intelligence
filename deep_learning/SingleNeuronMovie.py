import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

'''
Explanation of the Steps
Create a Mock Dataset: We simulate a small dataset of movies with features like genre, release year, runtime, 
and the target variable (rating).
Preprocess the Data: We encode categorical variables using one-hot encoding to convert genres into a format 
suitable for the model. The target variable is separated from the features.
Split the Data: The dataset is split into training and testing sets to evaluate model performance.
Build the Model: A simple neural network is constructed with one input layer, two hidden layers, and one output layer.
Compile the Model: The model is compiled with a loss function suitable for regression (Mean Squared Error) 
and an optimizer (Adam).
Train the Model: The model is trained for 100 epochs, with verbose output to monitor the training process.
Evaluate the Model: The model's performance is evaluated on the test set using the specified metrics.
Make Predictions: Finally, predictions are made on the test data, and the predicted ratings are printed.
'''

# Step 1: Create a mock dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Release Year': [2010, 2015, 2010, 2018, 2015],
    'Runtime': [120, 95, 110, 150, 100],
    'Rating': [8.0, 7.5, 8.5, 6.5, 7.0]  # Target variable
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Convert categorical variables to numerical using one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(df[['Genre']])
genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['Genre']))

# Combine with the original DataFrame
df_encoded = pd.concat([df.drop(columns=['Movie', 'Genre', 'Rating']), genre_df], axis=1)

# Define features and target
X = df_encoded.values  # Features
y = df['Rating'].values  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the neural network model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  # Hidden layer
model.add(Dense(5, activation='relu'))  # Second hidden layer
model.add(Dense(1))  # Output layer for regression

# Step 5: Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Step 6: Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Step 7: Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {loss:.2f}')
print(f'Test MAE: {mae:.2f}')

# Step 8: Make predictions (optional)
predictions = model.predict(X_test)
print(f'Predicted Ratings: {predictions.flatten()}')
