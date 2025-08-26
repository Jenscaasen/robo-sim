#!/usr/bin/env python3
"""
HTTP client for interacting with the URDF simulator API.
Handles joint control and camera image retrieval.
"""

import httpx
import base64
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

class SimulatorClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.base_url = f"http://{host}:{port}"
        self.client = httpx.AsyncClient()

    async def get_joints(self) -> List[Dict]:
        """Get current joint states from simulator"""
        response = await self.client.get(f"{self.base_url}/api/joints")
        response.raise_for_status()
        return response.json()

    async def set_joint(self, joint_id: int, value: float) -> Dict:
        """Set a single joint position"""
        response = await self.client.get(f"{self.base_url}/api/joint/{joint_id}/{value}")
        response.raise_for_status()
        return response.json()

    async def set_multiple_joints(self, joints: List[Dict], fast: bool = False) -> Dict:
        """Set multiple joint positions simultaneously"""
        endpoint = "/api/joints/instant" if not fast else "/api/joints/fast"
        response = await self.client.post(f"{self.base_url}{endpoint}", json=joints)
        response.raise_for_status()
        return response.json()

    async def reset_joints(self) -> Dict:
        """Reset all joints to neutral position"""
        response = await self.client.get(f"{self.base_url}/api/reset/instant")
        response.raise_for_status()
        return response.json()

    async def read_cam_image(self, camera_id: int) -> Dict:
        """Read image from specified camera and display it to the user"""
        response = await self.client.get(f"{self.base_url}/api/camera/{camera_id}")
        response.raise_for_status()

        # Return both the confirmation message and the image data
        return {
            "message": f"image for camera {camera_id} shown to user in chat window",
            "image_data": base64.b64encode(response.content).decode('utf-8')
        }

    def _decode_image(self, image_bytes) -> Optional[np.ndarray]:
        """Decode base64 image bytes to OpenCV BGR image"""
        try:
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Image decode error: {e}")
            return None

    def _detect_color_in_image(self, image: np.ndarray, color_name: str, camera_id: int) -> Optional[Tuple[int, int]]:
        """Detect a specific color in an image using HSV color detection"""
        if image is None:
            return None

        # Convert BGR to HSV for robust color thresholding
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges (HSV format)
        color_ranges = {
            'red': [
                (np.array([0, 70, 60], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
                (np.array([170, 70, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
            ],
            'blue': [
                (np.array([100, 70, 60], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
            ],
            'green': [
                (np.array([40, 70, 60], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8))
            ],
            'yellow': [
                (np.array([20, 70, 60], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
            ],
            'orange': [
                (np.array([10, 70, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
            ],
            'purple': [
                (np.array([140, 70, 60], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8))
            ]
        }

        if color_name.lower() not in color_ranges:
            return None

        # Create mask for the specified color
        masks = []
        for lower, upper in color_ranges[color_name.lower()]:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)

        if len(masks) == 1:
            color_mask = masks[0]
        else:
            # For colors with multiple ranges (like red), combine them
            color_mask = cv2.bitwise_or(masks[0], masks[1])

        # Morphological operations to reduce noise and fill small gaps
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Choose the largest contour (assuming the object is the dominant colored object)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 20:  # reject tiny blobs
            return None

        # Compute centroid
        M = cv2.moments(largest)
        if M.get("m00", 0) == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return cx, cy

    async def detect_color_position(self, color_name: str) -> Dict:
        """Detect the position of a specific color across all cameras"""
        try:
            positions = {}
            found_any = False

            # Capture images from all 3 cameras
            for camera_id in [1, 2, 3]:
                response = await self.client.get(f"{self.base_url}/api/camera/{camera_id}")
                response.raise_for_status()

                # Decode the image
                image = self._decode_image(response.content)
                if image is None:
                    positions[f"camera_{camera_id}"] = None
                    continue

                # Detect the color in this image
                position = self._detect_color_in_image(image, color_name, camera_id)
                positions[f"camera_{camera_id}"] = position

                if position is not None:
                    found_any = True

            if not found_any:
                return {
                    "message": f"No {color_name} objects detected in any camera",
                    "positions": positions
                }

            # Format the response
            message_parts = []
            pixel_coordinates = []

            for camera_id in [1, 2, 3]:
                pos = positions[f"camera_{camera_id}"]
                if pos is not None:
                    x, y = pos
                    message_parts.append(f"Camera {camera_id}: {color_name} at ({x}, {y})")
                    pixel_coordinates.extend([float(x), float(y)])
                else:
                    message_parts.append(f"Camera {camera_id}: {color_name} not found")
                    pixel_coordinates.extend([None, None])

            return {
                "message": " | ".join(message_parts),
                "positions": positions,
                "pixel_coordinates": pixel_coordinates
            }

        except Exception as e:
            return {
                "error": f"Failed to detect {color_name}: {str(e)}",
                "positions": None
            }

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()