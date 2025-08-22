import cv2
import numpy as np
import json
import os
import csv
import glob
import argparse

def find_yellow_region(image):
    """
    Find the yellow region in the image to use as reference point.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def get_black_blob_tip(image_path):
    """
    Get the tip position of the black blob (furthest from yellow region).
    """
    if not os.path.exists(image_path):
        return None, None
    
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create binary mask for black regions
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find largest black contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find yellow region
    yellow_center = find_yellow_region(image)
    
    # Find tip point furthest from yellow
    tip_point = None
    max_distance = 0
    
    if yellow_center:
        yellow_x, yellow_y = yellow_center
        
        for point in largest_contour:
            px, py = int(point[0][0]), int(point[0][1])
            distance = np.sqrt((px - yellow_x)**2 + (py - yellow_y)**2)
            
            if distance > max_distance:
                max_distance = distance
                tip_point = (px, py)
    else:
        # Fallback: rightmost point
        rightmost_idx = largest_contour[:, :, 0].argmax()
        rightmost_point = largest_contour[rightmost_idx][0]
        tip_point = (int(rightmost_point[0]), int(rightmost_point[1]))
    
    return tip_point if tip_point else (None, None)

def load_joint_positions(json_path):
    """
    Load joint positions from JSON file.
    """
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert list of dicts to dict
    positions = {}
    for item in data:
        positions[item['name']] = item['position']
    
    return positions

def compile_dataset(folder_name):
    """
    Compile dataset from folder containing JSON and image files.
    """
    if not os.path.exists(folder_name):
        print(f"Error: Folder {folder_name} does not exist")
        return
    
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_name, "datarow-*.json"))
    json_files.sort()
    
    if not json_files:
        print(f"No JSON files found in {folder_name}")
        return
    
    # Prepare CSV data
    csv_data = []
    
    # CSV header
    header = [
        'shoulder_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll',
        'cam1_tip_x', 'cam1_tip_y', 'cam2_tip_x', 'cam2_tip_y', 'cam3_tip_x', 'cam3_tip_y'
    ]
    
    print(f"Processing {len(json_files)} data rows...")
    
    for json_file in json_files:
        # Extract datarow number from filename
        basename = os.path.basename(json_file)
        datarow_num = basename.replace('datarow-', '').replace('.json', '')
        
        print(f"Processing datarow {datarow_num}...")
        
        # Load joint positions
        joint_positions = load_joint_positions(json_file)
        
        # Initialize row data
        row_data = []
        
        # Add joint positions
        for joint_name in ['shoulder_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll']:
            row_data.append(joint_positions.get(joint_name, ''))
        
        # Process camera images
        for cam_num in [1, 2, 3]:
            image_path = os.path.join(folder_name, f"datarow-{datarow_num}-cam{cam_num}.jpg")
            tip_x, tip_y = get_black_blob_tip(image_path)
            
            if tip_x is not None and tip_y is not None:
                row_data.extend([tip_x, tip_y])
                print(f"  cam{cam_num}: tip at ({tip_x}, {tip_y})")
            else:
                row_data.extend(['', ''])
                print(f"  cam{cam_num}: tip not detected")
        
        csv_data.append(row_data)
    
    # Write CSV file
    csv_filename = f"{folder_name}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(header)
        writer.writerows(csv_data)
    
    print(f"\nDataset compiled successfully!")
    print(f"Output file: {csv_filename}")
    print(f"Rows processed: {len(csv_data)}")

def main():
    parser = argparse.ArgumentParser(description='Compile dataset from folder with JSON and image files')
    parser.add_argument('folder_name', help='Name of the folder containing the dataset')
    
    args = parser.parse_args()
    compile_dataset(args.folder_name)

if __name__ == "__main__":
    # If no command line arguments, use default folder for testing
    import sys
    if len(sys.argv) == 1:
        print("No folder specified, using default: dataset_20250821_115559")
        compile_dataset("dataset_20250821_115559")
    else:
        main()