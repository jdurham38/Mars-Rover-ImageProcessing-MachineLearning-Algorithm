import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Step 1: Load and preprocess the image
image = cv2.imread('Mars_Perseverance_NRF_0757_0734152114_178ECM_N0372562NCAM00709_04_095J.png')
cv2.imshow('Original Image', image)
cv2.waitKey(0)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', image)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('grayscale_image.jpg', image)

image = cv2.equalizeHist(image)
cv2.imshow('Equalized Image', image)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('equalized_image.jpg', image)

# Detect edges using Canny algorithm and show the result
smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)
edges = cv2.Canny(smoothed_image, 100, 200)
cv2.imshow('Canny Edges', edges)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('canny_edges.jpg', edges)

# Localize edges using the Laplacian of Gaussian (LoG) method and show the result
log = cv2.GaussianBlur(image, (3, 3), 0)
log = cv2.Laplacian(log, cv2.CV_64F)
log = cv2.normalize(log, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
log[log < 0.5] = 0
kernel = np.ones((3, 3), np.uint8)

cv2.imshow('Laplacian of Gaussian Edges', edges)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('log_edges.jpg', edges)



# Show the result of closing and dilation
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
dilation = cv2.dilate(edges, kernel)
cv2.imshow('Closing', closing)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('closing.jpg', closing)

cv2.imshow('Dilation', dilation)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('dilation.jpg', dilation)

# Feature extraction using LBP and show the result
P = 9
R = 4
N_BINS = P * R
LBP_LENGTH = N_BINS * 2169
lbp = local_binary_pattern(image, P, R)
features = np.histogram(lbp.ravel(), bins=N_BINS, range=(0, N_BINS))[0]
if len(features) < LBP_LENGTH:
    features = np.concatenate([features, np.zeros(LBP_LENGTH - len(features))])
elif len(features) > LBP_LENGTH:
    features = features[:LBP_LENGTH]
cv2.imshow('Local Binary Pattern Features', lbp)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('lbp_features.jpg', lbp)

cv2.destroyAllWindows()
