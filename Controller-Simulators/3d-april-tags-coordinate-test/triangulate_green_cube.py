import os
import sys
import json
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# Add dataset compiler utils to path for calibration and triangulation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Dataset-Preparation", "2.dataset_compiler"))
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

try:
    from triangulation_utils import calibrate_cameras_for_folder, triangulate_tip
except Exception as e:
    print("Failed to import triangulation_utils. Ensure the path is correct and dependencies are installed.", file=sys.stderr)
    raise


def detect_green_centroid(bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect the largest green region and return its centroid (u, v) in pixels.
    Returns None if not found.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Green range (tuned for typical bright green in PyBullet renders)
    # Adjust if needed: view debug windows or save masks.
    lower1 = np.array([35, 40, 40], dtype=np.uint8)
    upper1 = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:  # too small
        return None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return (cx, cy)


def annotate_and_save(image_path: str, uv: Optional[Tuple[float, float]], out_suffix: str = "_annot") -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    if uv is not None:
        u, v = uv
        cv2.circle(img, (int(round(u)), int(round(v))), 8, (0, 0, 255), -1)
        cv2.putText(img, f"({int(round(u))},{int(round(v))})", (int(u) + 10, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out_path = os.path.splitext(image_path)[0] + out_suffix + os.path.splitext(image_path)[1]
    cv2.imwrite(out_path, img)


def main():
    folder = SCRIPT_DIR
    tags_json = os.path.join(folder, "tags_world.json")
    for p in [
        os.path.join(folder, "datarow-00001-cam1.jpg"),
        os.path.join(folder, "datarow-00001-cam2.jpg"),
        os.path.join(folder, "datarow-00001-cam3.jpg"),
        tags_json,
    ]:
        if not os.path.isfile(p):
            print(f"Missing required file: {p}")
            return

    # Calibrate cameras using AprilTags and world tag mapping
    print("Calibrating cameras from AprilTags...")
    calibrations = calibrate_cameras_for_folder(folder, tags_json_path=tags_json, fov_y_deg=60.0, cam_ids=[1, 2, 3])
    if not calibrations:
        print("Calibration failed: no cameras calibrated. Ensure tags are visible and tag detectors are installed (pupil-apriltags or apriltag).")
        return
    for cam_id, calib in calibrations.items():
        w, h = calib["wh"]
        print(f"Calibrated cam{cam_id}: image size {w}x{h}")

    # Detect green cube pixels in each camera
    tip_pixels_by_cam: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for cam in [1, 2, 3]:
        img_path = os.path.join(folder, f"datarow-00001-cam{cam}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image: {img_path}")
            tip_pixels_by_cam[cam] = (None, None)
            continue

        uv = detect_green_centroid(img)
        tip_pixels_by_cam[cam] = uv if uv is not None else (None, None)
        print(f"cam{cam}: green centroid = {uv}")
        annotate_and_save(img_path, uv, out_suffix="_green")

    # Triangulate 3D from available views
    print("Triangulating 3D position...")
    X = triangulate_tip(tip_pixels_by_cam, calibrations)
    if X is None:
        print("Triangulation failed. Need at least two cameras with detected green centroid and successful calibration.")
        return

    x, y, z = X
    print(f"Estimated 3D position (world frame): X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

    # Save JSON result
    out_json = os.path.join(folder, "triangulated_green_cube.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "point_world": {"x": x, "y": y, "z": z},
                "pixels": {str(k): {"u": v[0], "v": v[1]} for k, v in tip_pixels_by_cam.items()},
                "cams_calibrated": sorted(list(calibrations.keys())),
            },
            f,
            indent=2,
        )
    print(f"Wrote result: {out_json}")
    print("Annotated images saved alongside originals with suffix '_green'.")


if __name__ == "__main__":
    main()