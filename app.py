import cv2
import numpy as np
import time

# Ensure NumPy 2.x compatibility
def check_numpy_version():
    version = np.__version__.split('.')
    if int(version[0]) < 2:
        raise ImportError("This script requires NumPy 2.x. Please upgrade NumPy.")

check_numpy_version()

# Initialize the camera
cap = cv2.VideoCapture(0)
time.sleep(3)

# Capturing the background (taking multiple frames to reduce noise)
background_frames = []
for _ in range(30):  # Capture multiple frames
    ret, frame = cap.read()
    if ret:
        background_frames.append(frame)
background = np.median(np.array(background_frames), axis=0).astype(dtype=np.uint8)

# Flip background for consistency
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally
    frame = np.flip(frame, axis=1)
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for blue color (can be adjusted dynamically)
    lower_blue = np.array([90, 50, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    
    # Create mask to detect blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Refine mask using morphology
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Create inverse mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Extract parts of the image
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Merge the two to get final output
    output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    # Display output using OpenCV (instead of Matplotlib for efficiency)
    cv2.imshow("Invisibility Cloak", output)
    
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
