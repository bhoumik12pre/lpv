# Practical 7a
# Binary Classification using Deep Neural Network
# IMDB Movie Review Sentiment Analysis

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Vocabulary Size
vocab_size = 10000

# Load IMDB Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad Sequences
maxlen = 200

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build Deep Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(maxlen,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict Sample Reviews
predictions = model.predict(X_test[:5])

print("\nPredictions:")
for i, pred in enumerate(predictions):
    if pred > 0.5:
        print(f"Review {i+1}: Positive")
    else:
        print(f"Review {i+1}: Negative")