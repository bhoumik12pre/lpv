# Practical 8
# Fashion MNIST Classification using Deep Neural Network

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize Dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# Class Names
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Build Deep Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
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
    epochs=10,
    batch_size=128,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict First 5 Images
predictions = model.predict(X_test[:5])

print("\nPredicted Classes:")

for i in range(5):
    predicted_class = np.argmax(predictions[i])
    actual_class = y_test[i]

    print(f"Image {i+1}:")
    print("Predicted =", class_names[predicted_class])
    print("Actual    =", class_names[actual_class])
    print()