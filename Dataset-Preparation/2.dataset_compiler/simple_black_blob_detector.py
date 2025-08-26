import cv2
import numpy as np

def get_black_blob_position(image_path):
    """
    Simple function to get the black blob position in pixels.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (x, y) coordinates of the black blob center in pixels
    """
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask for black regions (threshold = 30)
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of black regions
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest black contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the centroid
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)

if __name__ == "__main__":
    image_path = "dataset_20250821_115559/datarow-00001-cam1.jpg"
    x, y = get_black_blob_position(image_path)
    print(f"Black blob position: ({x}, {y}) pixels")