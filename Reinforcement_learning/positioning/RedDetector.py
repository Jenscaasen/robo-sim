from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np


@dataclass
class Detection:
    found: bool
    cx: float  # pixel x
    cy: float  # pixel y
    area: float  # contour area in pixels
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _centroid_from_contour(contour: np.ndarray) -> Optional[Tuple[float, float, float, Tuple[int, int, int, int]]]:
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    if area < 5.0:
        return None
    M = cv2.moments(contour)
    if abs(M.get("m00", 0.0)) < 1e-6:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return cx, cy, area, (int(x), int(y), int(w), int(h))


def _mask_hsv_ranges(hsv: np.ndarray, ranges: Tuple[Tuple[np.ndarray, np.ndarray], ...]) -> np.ndarray:
    agg = None
    for lower, upper in ranges:
        part = cv2.inRange(hsv, lower, upper)
        agg = part if agg is None else cv2.bitwise_or(agg, part)
    return agg if agg is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)


class RedPurpleDetector:
    """
    HSV-based detector for red cube and purple gripper tip.
    - Red uses dual hue bands near 0 and 180.
    - Purple uses a single band tuned for the gripper color.

    Methods return normalized coordinates (0..1) if requested by the caller
    (the class outputs pixel coordinates; normalization is trivial using image shape).
    """

    def __init__(
        self,
        red_ranges: Optional[Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]] = None,
        purple_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
        morph_kernel: int = 3,
        morph_iters: int = 1,
        min_area: float = 30.0,
    ):
        # Defaults tuned for typical simulator renderings; allow caller to override.
        if red_ranges is None:
            red_ranges = (
                ((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (180, 255, 255)),
            )
        if purple_range is None:
            purple_range = ((125, 80, 80), (155, 255, 255))

        self.red_ranges = tuple(
            (np.array(l, dtype=np.uint8), np.array(u, dtype=np.uint8)) for l, u in red_ranges
        )
        self.purple_lower = np.array(purple_range[0], dtype=np.uint8)
        self.purple_upper = np.array(purple_range[1], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.morph_iters = int(morph_iters)
        self.min_area = float(min_area)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.morph_iters)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.morph_iters)
        return mask

    def detect_colors(self, bgr_image: np.ndarray) -> Dict[str, Detection]:
        """
        Returns a dict with entries:
        {
            "red": Detection(...),
            "purple": Detection(...)
        }
        """
        if bgr_image is None or bgr_image.size == 0:
            return {
                "red": Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0)),
                "purple": Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0)),
            }

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Red: combine two ranges
        red_mask = _mask_hsv_ranges(hsv, self.red_ranges)
        red_mask = self._postprocess(red_mask)
        red_contour = _largest_contour(red_mask)
        red_det = Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0))
        red_centroid = _centroid_from_contour(red_contour)
        if red_centroid is not None:
            cx, cy, area, bbox = red_centroid
            if area >= self.min_area:
                red_det = Detection(True, cx, cy, area, bbox)

        # Purple: single range
        purple_mask = cv2.inRange(hsv, self.purple_lower, self.purple_upper)
        purple_mask = self._postprocess(purple_mask)
        purple_contour = _largest_contour(purple_mask)
        purple_det = Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0))
        purple_centroid = _centroid_from_contour(purple_contour)
        if purple_centroid is not None:
            cx, cy, area, bbox = purple_centroid
            if area >= self.min_area:
                purple_det = Detection(True, cx, cy, area, bbox)

        return {"red": red_det, "purple": purple_det}

    @staticmethod
    def normalize_point(cx: float, cy: float, width: int, height: int) -> Tuple[float, float]:
        if width <= 0 or height <= 0 or cx < 0 or cy < 0:
            return -1.0, -1.0
        return float(cx) / float(width), float(cy) / float(height)