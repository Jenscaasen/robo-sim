import cv2
import numpy as np

def find_yellow_region(image):
    """
    Find the yellow region in the image to use as reference point.
    
    Args:
        image: BGR image
        
    Returns:
        tuple: (x, y) center of yellow region, or None if not found
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color in HSV
    # Yellow hue is around 20-30 in OpenCV HSV
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create mask for yellow regions
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest yellow contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    
    return None

def get_black_blob_analysis(image_path):
    """
    Analyze black blob and find both center and tip furthest from yellow region.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Analysis results with center, tip, and other info
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
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    # Find yellow region
    yellow_center = find_yellow_region(image)
    
    # Find the point on the black blob contour that's furthest from yellow
    tip_point = None
    max_distance = 0
    
    if yellow_center:
        yellow_x, yellow_y = yellow_center
        
        # Check each point on the contour
        for point in largest_contour:
            px, py = point[0]
            # Calculate distance from this point to yellow center
            distance = np.sqrt((px - yellow_x)**2 + (py - yellow_y)**2)
            
            if distance > max_distance:
                max_distance = distance
                tip_point = (px, py)
    else:
        # If no yellow found, find the rightmost point (assuming tip is on the right)
        rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        tip_point = rightmost_point
    
    # Create visualization
    result_image = image.copy()
    
    # Draw black blob contour
    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)
    
    # Mark center
    cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(result_image, f"Center ({center_x}, {center_y})", 
               (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Mark tip
    if tip_point:
        cv2.circle(result_image, tip_point, 5, (255, 0, 0), -1)
        cv2.putText(result_image, f"Tip ({tip_point[0]}, {tip_point[1]})", 
                   (tip_point[0] + 10, tip_point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Mark yellow center if found
    if yellow_center:
        cv2.circle(result_image, yellow_center, 5, (0, 255, 255), -1)
        cv2.putText(result_image, f"Yellow ({yellow_center[0]}, {yellow_center[1]})", 
                   (yellow_center[0] + 10, yellow_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Save result
    result_path = image_path.replace('.jpg', '_enhanced_detected.jpg')
    cv2.imwrite(result_path, result_image)
    
    return {
        'center': (center_x, center_y),
        'tip': tip_point,
        'yellow_center': yellow_center,
        'blob_area': cv2.contourArea(largest_contour),
        'distance_tip_to_yellow': max_distance if yellow_center else None,
        'result_image_path': result_path
    }

def get_black_blob_tip(image_path):
    """
    Simple function to get just the tip position.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (x, y) coordinates of the black blob tip furthest from yellow
    """
    result = get_black_blob_analysis(image_path)
    return result['tip']

if __name__ == "__main__":
    image_path = "dataset_20250821_115559/datarow-00001-cam1.jpg"
    
    print("=== Enhanced Black Blob Analysis ===")
    result = get_black_blob_analysis(image_path)
    
    print(f"Black blob center: {result['center']} pixels")
    print(f"Black blob tip (furthest from yellow): {result['tip']} pixels")
    print(f"Yellow region center: {result['yellow_center']} pixels")
    print(f"Black blob area: {result['blob_area']} pixels")
    if result['distance_tip_to_yellow']:
        print(f"Distance from tip to yellow: {result['distance_tip_to_yellow']:.1f} pixels")
    print(f"Enhanced visualization saved to: {result['result_image_path']}")