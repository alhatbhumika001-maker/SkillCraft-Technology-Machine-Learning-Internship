# Import libraries
import os
print(os.listdir())
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("house_price_dataset.csv")

# Display dataset
print(data.head())

# Select features (input)
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]

# Target (output)
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Plot results
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction using Linear Regression")
plt.show()