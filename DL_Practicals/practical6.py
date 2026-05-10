# Practical 6
# Linear Regression using Deep Neural Network
# Boston Housing Price Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Features and Target
X = data.drop("medv", axis=1).values
y = data["medv"].values

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=1
)

# Evaluate Model
loss, mae = model.evaluate(X_test, y_test)

print("\nMean Absolute Error:", mae)

# Predictions
predictions = model.predict(X_test[:5])

print("\nPredicted House Prices:")
print(predictions)

print("\nActual House Prices:")
print(y_test[:5])