import os
import json
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


def _try_import_apriltag_detector():
    """
    Try to import an AprilTag detector implementation, preferring pupil_apriltags.
    Returns a callable detect(gray: np.ndarray) -> List[dict{id:int, corners:np.ndarray(4,2)}]
    """
    # Prefer robust parameter-sweeping detector
    try:
        from tag_detection import get_detector  # type: ignore
        return get_detector()
    except Exception:
        pass

    # Fallback: pupil_apriltags (single config tuned for small/rendered tags)
    try:
        from pupil_apriltags import Detector  # type: ignore

        def detect_with_pupil(gray: np.ndarray) -> List[dict]:
            det = Detector(
                families="tag36h11,tagStandard41h12,tag25h9,tag16h5",
                nthreads=1,
                quad_decimate=0.5,
                quad_sigma=0.8,
                refine_edges=True,
                decode_sharpening=0.5,
            )
            results = det.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            out = []
            for r in results:
                out.append({"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)})
            return out

        return detect_with_pupil
    except Exception:
        pass

    # Fallback: python-apriltag (single config tuned for small/rendered tags)
    try:
        import apriltag  # type: ignore

        def detect_with_apriltag(gray: np.ndarray) -> List[dict]:
            options = apriltag.DetectorOptions(
                families="tag36h11",
                border=1,
                nthreads=1,
                quad_decimate=0.5,
                quad_blur=0.8,
                refine_edges=True,
                refine_decode=True,
                refine_pose=False,
                quad_contours=True,
            )
            atd = apriltag.Detector(options)
            results = atd.detect(gray)
            out = []
            for r in results:
                out.append({"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)})
            return out

        return detect_with_apriltag
    except Exception:
        pass

    raise ImportError(
        "No AprilTag detector available. Please install one of:\n"
        "  pip install pupil-apriltags\n"
        "or\n"
        "  pip install apriltag"
    )


def load_tags_world_json(path: str) -> Dict[int, np.ndarray]:
    """
    Load tags_world.json produced by the simulator.
    Returns: dict id -> corners_world [4x3 float32] in order [tl, tr, br, bl]
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"tags_world.json not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    tags_map: Dict[int, np.ndarray] = {}
    for t in data.get("tags", []):
        tag_id = int(t["id"])
        corners = np.array(t["corners_world"], dtype=np.float32)  # shape (4,3)
        if corners.shape != (4, 3):
            continue
        tags_map[tag_id] = corners
    return tags_map


def compute_intrinsics_from_fovy(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    """
    Compute pinhole intrinsics from vertical FOV (PyBullet's computeProjectionMatrixFOV uses vertical fov).
    fx = fy * aspect; fy = (H/2) / tan(FOVy/2)
    """
    fovy_rad = float(fov_y_deg) * np.pi / 180.0
    fy = (height * 0.5) / np.tan(0.5 * fovy_rad)
    fx = fy * (width / float(height))
    cx = width * 0.5
    cy = height * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def estimate_camera_pose_from_tags(
    K: np.ndarray,
    detections: List[dict],
    tags_world: Dict[int, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate camera pose (R, t) in the world frame from detected AprilTags.
    Input:
      - K: 3x3 intrinsics
      - detections: list of dict {'id': int, 'corners': (4x2) [tl,tr,br,bl]}
      - tags_world: dict id -> (4x3) corners_world [tl,tr,br,bl]
    Output:
      - (R, t) where x_cam = R * X_world + t
    """
    obj_pts = []
    img_pts = []
    for det in detections:
        tid = int(det["id"])
        if tid not in tags_world:
            continue
        cw = tags_world[tid]  # (4,3)
        uv = det["corners"]   # (4,2)
        if cw.shape != (4, 3) or uv.shape != (4, 2):
            continue
        # Accumulate all 4 corner correspondences
        for i in range(4):
            obj_pts.append(cw[i])
            img_pts.append(uv[i])

    if len(obj_pts) < 6:  # need at least 3 points, but prefer >=6 for stability
        return None

    obj = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.array(img_pts, dtype=np.float64).reshape(-1, 2)
    dist = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 3x4 projection matrix P = K [R|t] mapping world -> image (pixels).
    """
    Rt = np.hstack([R, t])
    return K @ Rt


def _find_first_image_for_cam(folder: str, cam_id: int) -> Optional[str]:
    """
    Find the first file matching '*-cam{cam_id}*.(jpg|jpeg|png)' in folder.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(folder, f"*-cam{cam_id}*{ext}")))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def calibrate_cameras_for_folder(
    folder: str,
    tags_json_path: str,
    fov_y_deg: float = 60.0,
    cam_ids: List[int] = [1, 2, 3],
) -> Dict[int, dict]:
    """
    Calibrate all listed cameras from a single representative image each.
    Returns dict cam_id -> {'K':3x3, 'R':3x3, 't':3x1, 'P':3x4, 'wh':(w,h)}
    """
    detector = _try_import_apriltag_detector()
    tags_world = load_tags_world_json(tags_json_path)

    calib: Dict[int, dict] = {}
    for cam in cam_ids:
        img_path = _find_first_image_for_cam(folder, cam)
        if not img_path or not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        K = compute_intrinsics_from_fovy(w, h, fov_y_deg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)
        pose = estimate_camera_pose_from_tags(K, dets, tags_world)
        if pose is None:
            # Try a tiny blur to help detector in case of aliasing
            gray2 = cv2.GaussianBlur(gray, (3, 3), 0.5)
            dets = detector(gray2)
            pose = estimate_camera_pose_from_tags(K, dets, tags_world)
        if pose is None:
            # Skip camera if PnP fails
            continue
        R, t = pose
        P = build_projection_matrix(K, R, t)
        calib[cam] = {"K": K, "R": R, "t": t, "P": P, "wh": (w, h)}
    return calib


def _triangulate_two(P1: np.ndarray, P2: np.ndarray, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    Triangulate a single 3D point from two views using OpenCV.
    Inputs:
      - P1, P2: 3x4 projection matrices
      - pt1, pt2: (u, v) pixel coordinates
    Returns:
      - X_world: np.ndarray shape (3,), or None on failure
    """
    p1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    p2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, p1, p2)  # 4x1
    w = X_h[3, 0]
    if abs(w) < 1e-9:
        return None
    X = (X_h[:3, 0] / w).reshape(3)
    return X


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    Project 3D world point X to pixel coordinates using K, R, t.
    """
    Xw = X.reshape(3, 1)
    xc = R @ Xw + t
    uvs = K @ xc
    u = float(uvs[0, 0] / uvs[2, 0])
    v = float(uvs[1, 0] / uvs[2, 0])
    return u, v


def _reprojection_error(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray, pt: Tuple[float, float]) -> float:
    u, v = _project(K, R, t, X)
    du = u - pt[0]
    dv = v - pt[1]
    return float(du * du + dv * dv)


def triangulate_tip(
    tip_pixels_by_cam: Dict[int, Tuple[Optional[float], Optional[float]]],
    calibrations: Dict[int, dict],
) -> Optional[Tuple[float, float, float]]:
    """
    Triangulate a 3D point from 2D observations across 2-3 calibrated cameras.
    Strategy:
      - Collect all valid cameras with both calibration and a detected tip.
      - If 2 cams: triangulate directly.
      - If 3 cams: triangulate for all three pairs, then pick the X with smallest total reprojection error.
    Returns:
      (x, y, z) in world coordinates, or None if not solvable.
    """
    # Collect eligible
    cams = []
    for cam_id, tip in tip_pixels_by_cam.items():
        if cam_id not in calibrations:
            continue
        tx, ty = tip
        if tx is None or ty is None:
            continue
        cams.append((cam_id, (float(tx), float(ty))))
    if len(cams) < 2:
        return None

    # Build matrices
    entries = []
    for cam_id, (u, v) in cams:
        c = calibrations[cam_id]
        entries.append((cam_id, c["P"], c["K"], c["R"], c["t"], (u, v)))

    # Two-view direct
    def tri_from_pair(a, b) -> Optional[np.ndarray]:
        _, P1, K1, R1, t1, pt1 = a
        _, P2, K2, R2, t2, pt2 = b
        X = _triangulate_two(P1, P2, pt1, pt2)
        return X

    best_X = None
    best_err = float("inf")

    if len(entries) == 2:
        X = tri_from_pair(entries[0], entries[1])
        if X is None:
            return None
        # Evaluate reprojection error over both views
        total = 0.0
        for _, _, K, R, t, pt in entries:
            total += _reprojection_error(K, R, t, X, pt)
        return (float(X[0]), float(X[1]), float(X[2]))

    # Three cams: evaluate all pairs
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            X = tri_from_pair(entries[i], entries[j])
            if X is None:
                continue
            total = 0.0
            for _, _, K, R, t, pt in entries:
                total += _reprojection_error(K, R, t, X, pt)
            if total < best_err:
                best_err = total
                best_X = X

    if best_X is None:
        return None
    return (float(best_X[0]), float(best_X[1]), float(best_X[2]))