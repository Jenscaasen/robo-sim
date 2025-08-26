#!/usr/bin/env python3
# Computer vision helpers: color segmentation and centroid extraction
# Keeps each function small and composable to stay < 300 lines.

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# -------------------------------
# HSV thresholds (tweakable)
# -------------------------------
# Each entry: List of (lower, upper) HSV bounds; multiple ranges are OR'ed.
HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    # Red wraps hue around 0
    "red": [
        (np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ],
    # Purple gripper (tune if needed)
    # Simulator tones usually sit in H ~ 130-155 (magenta/purple)
    "purple": [
        (np.array([125, 60, 60], dtype=np.uint8), np.array([155, 255, 255], dtype=np.uint8)),
    ],
    # Optional masks for blue/green etc. can be added here
}


def _make_mask_hsv(image_bgr: np.ndarray, ranges: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Combine multiple HSV intervals into one binary mask."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _largest_contour_centroid(mask: np.ndarray, min_area: float = 30.0) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
    """
    Return (cx, cy, area, bbox) for largest contour, or None.
    bbox = (x, y, w, h)
    """
    # Clean up a bit to stabilize centroid
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area:
        return None
    M = cv2.moments(c)
    if abs(M.get("m00", 0.0)) < 1e-6:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(c)
    return cx, cy, area, (x, y, w, h)


def detect_color_centroid(image_bgr: np.ndarray, color_name: str, min_area: float = 30.0) -> Optional[Tuple[int, int]]:
    """
    Generic color centroid detector using predefined HSV_RANGES.
    Returns (cx, cy) in pixels, or None.
    """
    ranges = HSV_RANGES.get(color_name.lower())
    if not ranges:
        raise ValueError(f"Unknown color '{color_name}'. Available: {list(HSV_RANGES.keys())}")
    mask = _make_mask_hsv(image_bgr, ranges)
    res = _largest_contour_centroid(mask, min_area=min_area)
    if res is None:
        return None
    cx, cy, _, _ = res
    return (cx, cy)


def detect_with_debug(image_bgr: np.ndarray, color_name: str, min_area: float = 30.0) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    """
    Detect color centroid and return (pos, debug_overlay).
    """
    ranges = HSV_RANGES.get(color_name.lower())
    if not ranges:
        raise ValueError(f"Unknown color '{color_name}'. Available: {list(HSV_RANGES.keys())}")
    mask = _make_mask_hsv(image_bgr, ranges)
    overlay = image_bgr.copy()
    pos = None
    res = _largest_contour_centroid(mask, min_area=min_area)
    if res is not None:
        cx, cy, area, (x, y, w, h) = res
        pos = (cx, cy)
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(overlay, f"{color_name} area={int(area)}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
    return pos, overlay


def detect_color_in_all_cameras(images_bgr: List[np.ndarray], color_name: str, min_area: float = 30.0) -> Optional[List[Tuple[int, int]]]:
    """
    Detect (cx, cy) in each image for the given color. Returns None if any view fails.
    """
    if not images_bgr or len(images_bgr) != 3:
        return None
    out: List[Tuple[int, int]] = []
    for img in images_bgr:
        pos = detect_color_centroid(img, color_name, min_area=min_area)
        if pos is None:
            return None
        out.append(pos)
    return out


def save_debug_overlays(images_bgr: List[np.ndarray], color_name: str, path_prefix: str) -> List[Optional[Tuple[int, int]]]:
    """
    Save debug overlays for each camera to help tuning HSV ranges.
    Returns list of positions (can contain None).
    """
    positions: List[Optional[Tuple[int, int]]] = []
    for i, img in enumerate(images_bgr):
        pos, overlay = detect_with_debug(img, color_name)
        cv2.imwrite(f"{path_prefix}_cam{i+1}_{color_name}.jpg", overlay)
        positions.append(pos)
    return positions


# Convenience aliases
def detect_red_cube_in_all_cameras(images_bgr: List[np.ndarray]) -> Optional[List[Tuple[int, int]]]:
    return detect_color_in_all_cameras(images_bgr, "red")


def detect_purple_gripper_in_all_cameras(images_bgr: List[np.ndarray]) -> Optional[List[Tuple[int, int]]]:
    return detect_color_in_all_cameras(images_bgr, "purple")