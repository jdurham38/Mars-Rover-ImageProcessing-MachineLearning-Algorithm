import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.linear_model import SGDClassifier
import pickle

# Load the classifier from the pickle file
with open('classifier.pkl', 'rb') as f:
    data = pickle.load(f)
    if isinstance(data, tuple):
        clf, class_set = data
    else:
        clf = data
        class_set = set()


# Define the preprocess_image and detect_edges functions

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img_resized = cv2.resize(img, (100, 100))  # resize the image to 100 x 100 pixels
    return img_resized


def detect_edges(image):
    # Step 1: Apply Gaussian blur to smooth the image and reduce noise
    smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Step 2: Detect edges using Canny algorithm
    edges = cv2.Canny(smoothed_image, 100, 200)

    # Step 3: Localize edges using the Laplacian of Gaussian (LoG) method
    log = cv2.GaussianBlur(image, (3, 3), 0)  # apply another Gaussian blur
    log = cv2.Laplacian(log, cv2.CV_64F)
    log = cv2.normalize(log, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    log[log < 128] = 0  # threshold to remove weak edges
    log = cv2.dilate(log, None)  # dilate edges for better visualization

    # Concatenate the Canny edges and Laplacian of Gaussian edges
    edges = np.concatenate([edges, log], axis=1)

    # Flatten the edges array to a 1D array
    flattened_edges = edges.flatten()

    # If the flattened array has fewer elements than max_length, pad it with zeros
    max_length = 78084
    if len(flattened_edges) < max_length:
        padding_length = max_length - len(flattened_edges)
        padding = np.zeros(padding_length)
        flattened_edges = np.concatenate([flattened_edges, padding])

    return flattened_edges


# Classify new images
img_paths = [
    'Mars_Perseverance_FRF_0749_0733447897_979ECM_N0370000FHAZ00203_10_195J.png']  # List of paths to new images
for img_path in img_paths:
    img = cv2.imread(img_path)
    img_processed = preprocess_image(img)
    edges_processed = detect_edges(img_processed)
    predicted_label = clf.predict([edges_processed])[0]
    print(f"Image {img_path} - Predicted label: {predicted_label}")
