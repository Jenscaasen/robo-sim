import requests
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


def build_base_url(host: str = "127.0.0.1", port: int = 5000) -> str:
    return f"http://{host}:{port}"


def get_current_joint_positions(base_url: str) -> Tuple[Optional[List[float]], Optional[Dict]]:
    """
    Fetch current joint data from backend and extract current angles for joints 0..4.
    Returns (positions_list, raw_json_dict) or (None, None) on failure.
    """
    try:
        response = requests.get(f"{base_url}/api/joints", timeout=5)
        response.raise_for_status()
        joints_data: Dict = response.json()

        current_positions: List[float] = []
        for i in range(5):
            key = str(i)
            if key in joints_data:
                joint = joints_data[key]
                current_positions.append(float(joint.get("current", 0.0)))
            else:
                current_positions.append(0.0)
        return current_positions, joints_data
    except requests.exceptions.RequestException as e:
        print(f"API error (/api/joints): {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error parsing joints: {e}")
        return None, None


def capture_camera_image(base_url: str, camera_id: int) -> Optional[np.ndarray]:
    """
    Capture image bytes from backend, decode to OpenCV BGR image.
    """
    try:
        resp = requests.get(f"{base_url}/api/camera/{camera_id}", timeout=10)
        resp.raise_for_status()
        image_bytes = resp.content
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Camera {camera_id}: decode failed")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Camera {camera_id} error: {e}")
        return None
    except Exception as e:
        print(f"Camera {camera_id} processing error: {e}")
        return None


def capture_all_cameras(base_url: str, camera_ids: Tuple[int, int, int] = (1, 2, 3)) -> Optional[List[np.ndarray]]:
    images: List[np.ndarray] = []
    for cid in camera_ids:
        img = capture_camera_image(base_url, cid)
        if img is None:
            return None
        images.append(img)
    return images


def send_joint_positions(base_url: str, joint_positions: List[float], joint_ids: List[int] = None) -> bool:
    """
    Send joint positions to the robot API.
    
    Args:
        base_url: Base URL of the API
        joint_positions: List of joint angles in radians
        joint_ids: List of joint IDs to set (default: [0, 1, 2, 3, 4])
        
    Returns:
        True if successful, False otherwise
    """
    if joint_ids is None:
        joint_ids = list(range(len(joint_positions)))
    
    if len(joint_positions) != len(joint_ids):
        print(f"Error: {len(joint_positions)} positions but {len(joint_ids)} joint IDs")
        return False
    
    success_count = 0
    total_joints = len(joint_ids)
    
    try:
        # Send GET request for each joint individually
        # Format: http://127.0.0.1:5000/api/joint/(number)/(angle)
        for joint_id, position in zip(joint_ids, joint_positions):
            url = f"{base_url}/api/joint/{joint_id}/{float(position)}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            success_count += 1
        
        if success_count == total_joints:
            print("✓ All joint positions sent successfully")
            return True
        else:
            print(f"⚠ Only {success_count}/{total_joints} joint positions sent successfully")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"API error setting joint positions: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error setting joint positions: {e}")
        return False