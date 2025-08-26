import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def find_red_cube_position(image: np.ndarray, camera_id: int, debug: bool = True, debug_dir: str = ".") -> Optional[Tuple[int, int]]:
    """
    Find the position of a red cube in an image using HSV color detection.

    Args:
        image: OpenCV image (BGR format)
        camera_id: Camera identifier used in debug outputs
        debug: Whether to save a debug overlay image
        debug_dir: Directory to save debug images

    Returns:
        (x, y) pixel coordinates of the detected red cube center, or None if not found
    """
    if image is None:
        return None

    # Convert BGR to HSV for robust color thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red wraps hue=0, so use two ranges and OR them
    # Tune S and V lower bounds to avoid picking low-saturation arm parts
    lower_red1 = np.array([0, 70, 60], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 60], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological ops to reduce noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Choose the largest red contour (assuming cube is dominant red object)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 20:  # reject tiny blobs
        return None

    # Optional shape heuristic: prefer roughly compact shapes (cube projection)
    # Compute bounding rect and aspect ratio if needed in the future.

    # Compute centroid
    M = cv2.moments(largest)
    if M.get("m00", 0) == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        overlay = image.copy()
        cv2.drawContours(overlay, [largest], -1, (0, 0, 255), 2)
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(overlay, f"({cx}, {cy}) area={int(area)}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out_path = os.path.join(debug_dir, f"debug_cam{camera_id}_red_detection.jpg")
        cv2.imwrite(out_path, overlay)

    return cx, cy


def detect_red_cube_in_all_cameras(images: List[np.ndarray], debug: bool = True, debug_dir: str = ".") -> Optional[List[Tuple[int, int]]]:
    """
    Detect the red cube in all camera images.

    Args:
        images: List of exactly 3 BGR images corresponding to cameras 1,2,3
        debug: Whether to save debug overlays
        debug_dir: Directory to save debug images

    Returns:
        List of (x, y) pixel coordinates for each camera [cam1, cam2, cam3], or None on failure
    """
    if not images or len(images) != 3:
        print("Invalid camera images list; expected 3 images")
        return None

    positions: List[Tuple[int, int]] = []
    for idx, img in enumerate(images):
        cam_id = idx + 1
        pos = find_red_cube_position(img, cam_id, debug=debug, debug_dir=debug_dir)
        if pos is None:
            print(f"Camera {cam_id}: red cube not found")
            return None
        positions.append(pos)

    return positions