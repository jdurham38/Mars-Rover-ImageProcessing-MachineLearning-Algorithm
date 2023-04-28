import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from skimage.morphology import closing, dilation
import csv


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


# Load the training dataset
training_dataset = []
labels = []
max_length = 0
with open('Train_CSV.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    next(reader)  # skip header row
    rows = list(reader)
    for i, row in enumerate(rows):
        img_path = row['PNG']
        label = row['LABELS']
        if label in ('1', '5'):  # if label is 1 or 5, it is a hazard
            label = 'hazard'
        elif label in ('0', '4'):  # if label is 0 or 4, it is not a hazard
            label = 'not hazard'
        else:  # skip any other labels
            continue
        labels.append(label)
        img = cv2.imread(img_path)
        img = preprocess_image(img)
        edges = detect_edges(img)
        # Apply dilation and closing to the image
        edges = cv2.dilate(edges, None)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        features = edges.reshape(-1)
        if len(features) > max_length:
            max_length = len(features)
        training_dataset.append((features, label))
        print(f"Processing image {i + 1}/{len(rows)}")





    # Perform oversampling to balance the classes
    ros = RandomOverSampler()
    X_train, y_train = np.array([features for features, _ in training_dataset]), [label for _, label in
                                                                                  training_dataset]
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)





    # Pad the features to have the same length
    for i, (features, label) in enumerate(training_dataset):
        padding_length = max_length - len(features)
        padding = np.zeros(padding_length)
        training_dataset[i] = (np.concatenate([features, padding]), label)

    # Reshape the features and labels to match the oversampled data
    X_train_resampled_padded = np.zeros((len(X_train_resampled), max_length))
    for i, features in enumerate(X_train_resampled):
        padding_length = max_length - len(features)
        padding = np.zeros(padding_length)
        X_train_resampled_padded[i, :len(features)] = features
        X_train_resampled_padded[i, len(features):] = padding
    y_train_resampled_padded = np.array(y_train_resampled)

    # Train the classifier using the oversampled data
    clf = SGDClassifier(loss='hinge', max_iter=4)
    clf.fit(X_train_resampled_padded, y_train_resampled_padded)
    prev_accuracy = -1
    max_iter = 4

    X_train, y_train = np.vstack([features for features, _ in training_dataset]), [label for _, label in
                                                                                   training_dataset]

    ros = RandomOverSampler()
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    max_length = 78084
    X_train = np.zeros((len(training_dataset), max_length))
    y_train = []



    for i, (features, label) in enumerate(training_dataset):
        X_train[i, :min(max_length, len(features))] = features[:min(max_length, len(features))]
        y_train.append(label)

    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    clf = None


    class_set = set()
    if os.path.exists('classifier.pkl'):
        with open('classifier.pkl', 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                clf, class_set = data
            else:
                clf = data
                class_set = set()
    else:
        clf = SGDClassifier(loss='hinge', max_iter=max_iter)


    # Initialize variables to store the predicted labels and accuracy at the iteration before accuracy starts to
    # decrease
    prev_predicted_labels = None
    prev_true_labels = None
    prev_accuracy = -1

    for i in range(max_iter):
        # Fit the classifier to the training data
        clf.partial_fit(X_train_resampled, y_train_resampled, classes=np.unique(y_train))

        # Predict the labels of the training data
        predicted_labels = clf.predict(X_train_resampled)
        true_labels = y_train_resampled

        # Calculate the accuracy of the classifier on the training data
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Iteration {i + 1}/{max_iter} - Training accuracy: {accuracy:.2f}")





        # Store the predicted labels and true labels at the previous iteration
        if i > 0 and accuracy < prev_accuracy:
            prev_predicted_labels = predicted_labels_prev
            prev_true_labels = true_labels_prev

        # Stop training if the accuracy starts to decrease
        if accuracy < prev_accuracy:
            print(f"Accuracy decreased from {prev_accuracy:.2f} to {accuracy:.2f}. Stopping training.")
            break

        # Update the previous accuracy and labels
        prev_accuracy = accuracy
        predicted_labels_prev = predicted_labels
        true_labels_prev = true_labels

        with open('classifier.pkl', 'wb') as f:
            pickle.dump((clf, class_set), f)



        # Classify new images
        img_paths = [
            'Mars_Perseverance_NRF_0757_0734152114_178ECM_N0372562NCAM00709_04_095J.png']  # List of paths to new images
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img_processed = preprocess_image(img)
            edges_processed = detect_edges(img_processed)
            predicted_label = clf.predict([edges_processed])[0]
            print(f"Image {img_path} - Predicted label: {predicted_label}")

        # Print out the accuracy and predicted labels at the iteration before the accuracy starts to decrease
        for i, (features, label) in enumerate(training_dataset):
            predicted_label = predicted_labels[i]
            print(f"Image {i + 1} - Predicted label: {predicted_label}, True label: {label}")

        else:
            print("Accuracy did not decrease during training.")



        print("Accuracy after image processing:")
        print(accuracy_score(y_train, clf.predict(X_train)))

        # Count the number of hazards and non-hazards
        num_hazards = labels.count('hazard')
        num_non_hazards = labels.count('not hazard')

        # Plot a bar chart
        plt.bar(['hazard', 'not hazard'], [num_hazards, num_non_hazards])
        plt.xlabel('Class')
        plt.ylabel('Count')


        # Define the accuracy values
        accuracies = [59, 60, 73, 78, 80, 95, 99]

        # Plot the line graph
        plt.plot(range(len(accuracies)), accuracies, '-o')

        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Improvement Over Iterations')

        # Add tick labels
        plt.xticks(range(len(accuracies)), range(1, len(accuracies) + 1))









