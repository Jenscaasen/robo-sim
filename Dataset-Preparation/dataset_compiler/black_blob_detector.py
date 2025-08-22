import cv2
import numpy as np
import os

def find_black_blob_position(image_path):
    """
    Find the position of a black blob in an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (x, y) coordinates of the black blob center in pixels, or None if not found
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask for black regions
    # Threshold for black pixels (adjust if needed)
    black_threshold = 30  # Pixels with intensity below this are considered black
    _, black_mask = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of black regions
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No black regions found in the image")
        return None
    
    # Find the largest black contour (assuming it's the main black blob)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the area to filter out noise
    area = cv2.contourArea(largest_contour)
    print(f"Largest black blob area: {area} pixels")
    
    # If the area is too small, it might be noise
    if area < 10:  # Minimum area threshold
        print("Black blob too small, might be noise")
        return None
    
    # Calculate the centroid of the largest black blob
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        print(f"Black blob center position: ({cx}, {cy}) pixels")
        
        # Optional: Create a visualization
        result_image = image.copy()
        cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(result_image, f"({cx}, {cy})", (cx + 10, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the result image
        result_path = image_path.replace('.jpg', '_detected.jpg')
        cv2.imwrite(result_path, result_image)
        print(f"Result image saved to: {result_path}")
        
        return (cx, cy)
    else:
        print("Could not calculate centroid of black blob")
        return None

def find_black_blob_alternative_method(image_path):
    """
    Alternative method using SimpleBlobDetector for black blob detection.
    """
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the image so black blobs become white (SimpleBlobDetector looks for white blobs)
    inverted = cv2.bitwise_not(gray)
    
    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 10000
    
    # Filter by Circularity
    params.filterByCircularity = False
    
    # Filter by Convexity
    params.filterByConvexity = False
    
    # Filter by Inertia
    params.filterByInertia = False
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(inverted)
    
    if keypoints:
        # Get the largest blob (by size)
        largest_blob = max(keypoints, key=lambda kp: kp.size)
        x, y = int(largest_blob.pt[0]), int(largest_blob.pt[1])
        
        print(f"Alternative method - Black blob center: ({x}, {y}) pixels")
        print(f"Blob size: {largest_blob.size}")
        
        return (x, y)
    else:
        print("Alternative method - No black blobs detected")
        return None

if __name__ == "__main__":
    # Path to the image
    image_path = "dataset_20250821_115559/datarow-00001-cam1.jpg"
    
    print("=== Black Blob Detection ===")
    print(f"Analyzing image: {image_path}")
    print()
    
    # Method 1: Contour-based detection
    print("Method 1: Contour-based detection")
    position1 = find_black_blob_position(image_path)
    print()
    
    # Method 2: SimpleBlobDetector
    print("Method 2: SimpleBlobDetector")
    position2 = find_black_blob_alternative_method(image_path)
    print()
    
    # Summary
    print("=== Summary ===")
    if position1:
        print(f"Contour method result: Black blob at ({position1[0]}, {position1[1]}) pixels")
    if position2:
        print(f"Blob detector method result: Black blob at ({position2[0]}, {position2[1]}) pixels")
    
    if not position1 and not position2:
        print("No black blob detected by either method")