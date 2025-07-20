from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# กำหนดมิติ Input สำหรับโมเดล LSTM
# INPUT_SHAPE = (SEQUENCE_LENGTH, จำนวนพิกัดของ Landmark)
# ในที่นี้คือ (30, 63)
INPUT_SHAPE = (SEQUENCE_LENGTH, 21 * 3) 
NUM_CLASSES = len(ACTIONS)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=INPUT_SHAPE),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, activation='relu'), # Last LSTM layer doesn't return sequences
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax') # Output layer for classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. การฝึกโมเดล (สมมติว่าคุณมี X_train, y_train, X_test, y_test แล้ว) ---
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[...])
# model.save('dynamic_sign_language_model.h5')