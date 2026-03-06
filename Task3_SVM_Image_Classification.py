# Import libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset path
dataset_path = "dataset"

categories = ["cats", "dogs"]

data = []
labels = []

# Load images
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)

            # Resize image
            image = cv2.resize(image, (64,64))

            # Convert image to array
            image = image.flatten()

            data.append(image)
            labels.append(label)

        except:
            print("Skipped:", img_path)

print("Dataset Loaded Successfully")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

print("Training Model...")

# Train model
model.fit(X_train, y_train)

print("Model Training Completed")

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)