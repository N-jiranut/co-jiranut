import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# Load your data
df = pd.read_csv("data/gpt_maindata_max.csv")

# Extract features and labels
features = df.iloc[:, :-1].values  # Columns 0 to 182
labels = df.iloc[:, -1].values     # Column 183 (pose class: 0–4)

# Create sequences (e.g., 30 frames per sample)
sequence_length = 30
X, y = [], []

for i in range(0, len(features) - sequence_length + 1, sequence_length):
    seq = features[i:i+sequence_length]
    label = labels[i+sequence_length-1]
    X.append(seq)
    y.append(label)

X = np.array(X)                        # Shape: (samples, 30, 183)
y = to_categorical(y, num_classes=5)  # One-hot encoding (5 classes)

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 183)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')  # 5 poses
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)

import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
