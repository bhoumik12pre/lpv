# Practical 7b
# Multiclass Classification using Deep Neural Network
# OCR Letter Recognition Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
data = pd.read_csv("letter-recognition.data", header=None)

# Features and Labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Deep Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(16,)),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict Sample Letters
predictions = model.predict(X_test[:5])

print("\nPredicted Classes:")
for i in range(5):
    predicted_letter = encoder.inverse_transform(
        [np.argmax(predictions[i])]
    )[0]

    actual_letter = encoder.inverse_transform([y_test[i]])[0]

    print(f"Sample {i+1}: Predicted = {predicted_letter}, Actual = {actual_letter}")