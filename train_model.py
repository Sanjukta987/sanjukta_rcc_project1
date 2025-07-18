# train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load and preprocess dataset
data = []
labels = []

for category in ['yes', 'no']:
    path = os.path.join("data", category)
    label = 1 if category == 'yes' else 0
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (100, 100))
            data.append(img_arr)
            labels.append(label)
        except:
            pass

data = np.array(data).reshape(-1, 100, 100, 1) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=16)
model.save("model/brain_tumor_model.h5")

print("✅ Model training completed and saved.")
