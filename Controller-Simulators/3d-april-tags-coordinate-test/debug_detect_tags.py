import os
import sys
from typing import List, Tuple
import cv2
import numpy as np

# Set paths to reuse triangulation utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Dataset-Preparation", "2.dataset_compiler"))
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

import triangulation_utils as tu  # [triangulation_utils.py](../../Dataset-Preparation/2.dataset_compiler/triangulation_utils.py:1)

def draw_tag_polygon(img: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int], tid: int) -> None:
    pts = corners.reshape(-1, 2).astype(int)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, p1, p2, color, 2)
    tl = tuple(pts[0])
    cv2.circle(img, tl, 4, (0, 0, 255), -1)
    cv2.putText(img, f"id={tid}", (tl[0] + 5, tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    print("Debugging AprilTag detections and PnP per camera")
    detector = tu._try_import_apriltag_detector()  # [triangulation_utils._try_import_apriltag_detector()](../../Dataset-Preparation/2.dataset_compiler/triangulation_utils.py:10)
    tags_json = os.path.join(SCRIPT_DIR, "tags_world.json")
    tags_world = tu.load_tags_world_json(tags_json)  # [triangulation_utils.load_tags_world_json()](../../Dataset-Preparation/2.dataset_compiler/triangulation_utils.py:71)

    for cam in [1, 2, 3]:
        img_path = os.path.join(SCRIPT_DIR, f"datarow-00001-cam{cam}.jpg")
        if not os.path.isfile(img_path):
            print(f"[cam{cam}] missing image {img_path}")
            continue

        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"[cam{cam}] failed to read image")
            continue

        h, w = img_color.shape[:2]
        K = tu.compute_intrinsics_from_fovy(w, h, 60.0)  # [triangulation_utils.compute_intrinsics_from_fovy()](../../Dataset-Preparation/2.dataset_compiler/triangulation_utils.py:91)
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # First pass
        dets = detector(gray)
        ids = [int(d["id"]) for d in dets]
        print(f"[cam{cam}] detections: {len(dets)} ids={ids}")

        # If nothing or too few for PnP, try slight blur
        if len(dets) < 2:
            gray2 = cv2.GaussianBlur(gray, (3, 3), 0.5)
            dets2 = detector(gray2)
            if len(dets2) > len(dets):
                dets = dets2
                ids = [int(d["id"]) for d in dets]
                print(f"[cam{cam}] after blur detections: {len(dets)} ids={ids}")

        # Annotate and save detections for visual inspection
        vis = img_color.copy()
        for d in dets:
            draw_tag_polygon(vis, d["corners"], (0, 255, 0), int(d["id"]))
        out_path = os.path.splitext(img_path)[0] + "_tags.jpg"
        cv2.imwrite(out_path, vis)
        print(f"[cam{cam}] wrote annotated: {out_path}")

        # Attempt PnP
        pose = tu.estimate_camera_pose_from_tags(K, dets, tags_world)  # [triangulation_utils.estimate_camera_pose_from_tags()](../../Dataset-Preparation/2.dataset_compiler/triangulation_utils.py:105)
        if pose is None:
            print(f"[cam{cam}] PnP: FAILED (need at least 2 tags with IDs present in tags_world.json)")
        else:
            R, t = pose
            print(f"[cam{cam}] PnP: OK. t={t.ravel()}")

if __name__ == "__main__":
    main()