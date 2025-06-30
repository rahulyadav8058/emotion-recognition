import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

from utils import load_fer2013_from_folder
DATASET_PATH = r'C:\Users\ry702\Desktop\fire\ml\emotion_recognition\dataset\archive\train'
X, y = load_fer2013_from_folder(DATASET_PATH)

# Load and preprocess dataset from image folders
X, y = load_fer2013_from_folder(DATASET_PATH)

print("Loaded images:", len(X))
print("Loaded labels:", len(y))

if len(X) == 0:
    print(" ERROR: No images loaded. Please check the dataset path and folder structure.")
    exit()

y_cat = to_categorical(y, num_classes=7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Save model
os.makedirs('model', exist_ok=True)
model.save('model/emotion_model.h5')
print(" Model saved to model/emotion_model.h5")
