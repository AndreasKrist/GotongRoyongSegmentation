import cv2
import numpy as np

# Callback function for trackbars
def update_canny(val):
    # Get current positions of sliders
    min_thresh = cv2.getTrackbarPos('Min Threshold', 'Edge Detection')
    max_thresh = cv2.getTrackbarPos('Max Threshold', 'Edge Detection')
    blur_ksize = cv2.getTrackbarPos('Blur Kernel Size', 'Edge Detection')
    
    # Ensure the kernel size is odd and at least 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize < 1:
        blur_ksize = 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, min_thresh, max_thresh)
    
    # Stack original and edges side by side for comparison
    combined = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('Edge Detection', combined)

# Load the image
image = cv2.imread('input_image.jpg')  # Replace with your image path
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a window
cv2.namedWindow('Edge Detection')

# Create trackbars for min and max thresholds, and blur kernel size
cv2.createTrackbar('Min Threshold', 'Edge Detection', 50, 255, update_canny)
cv2.createTrackbar('Max Threshold', 'Edge Detection', 150, 255, update_canny)
cv2.createTrackbar('Blur Kernel Size', 'Edge Detection', 1, 31, update_canny)

# Initial display
update_canny(0)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()
