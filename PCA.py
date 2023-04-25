import os
import cv2
import numpy as np
from sklearn.decomposition import PCA

# Path to image directory
img_dir = 'Dataset'

# Get list of image filenames, ignoring directories
img_filenames = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

# Load images and convert to grayscale
images = []
for filename in img_filenames:
    img = cv2.imread(os.path.join(img_dir, filename))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (100, 100))  # resize images to same dimensions
    images.append(img_resized)

# Stack images into a single array
images = np.stack(images)

# Flatten images
images_flat = images.reshape(images.shape[0], -1)

# Perform PCA with svd_solver='arpack'
n_components =5
pca = PCA(n_components=n_components, svd_solver='full')
pca.fit(images_flat)

# Transform images to reduced dimensionality
images_pca = pca.transform(images_flat)

# Reconstruct images from PCA components
images_reconstructed = pca.inverse_transform(images_pca)

# Reshape images to original dimensions
images_reconstructed = images_reconstructed.reshape(images.shape)

# Display original and reconstructed images
for i in range(images.shape[0]):
    cv2.imshow('Original', images[i])
    cv2.imshow('Reconstructed', images_reconstructed[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()