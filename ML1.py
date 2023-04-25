# Import library
import cv2
import os
import numpy as np

# Set path to dataset
dataset_path = 'Dataset'

# Define categories
categories = os.listdir(dataset_path)

# Create empty arrays for X_train and y_train
X_train = []
y_train = []
# Load images and labels
for img_file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_file)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (100, 100))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # ubah gambar menjadi gray
    X_train.append(gray_image)
    label = int(img_file.split('_')[0].replace('User', ''))
    y_train.append(label)

# Convert to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train).ravel()
unique_labels = np.unique(y_train)
X_train = np.reshape(X_train, (-1, 100, 100, 1))
# Save to file
np.save('label_encoder.npy', unique_labels)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)