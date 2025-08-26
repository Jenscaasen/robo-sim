import os
import sys
from typing import List, Tuple
import cv2
import numpy as np

# Path so we can import our utils and sweeper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Dataset-Preparation", "2.dataset_compiler"))
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# Optional: robust sweeper if import works
sweeper = None
try:
    import tag_detection as td  # [tag_detection.py](../../Dataset-Preparation/2.dataset_compiler/tag_detection.py:1)
    sweeper = td.get_detector()  # [tag_detection.get_detector()](../../Dataset-Preparation/2.dataset_compiler/tag_detection.py:3)
except Exception:
    sweeper = None

def print_env():
    import importlib, importlib.util
    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)
    for m in ["pupil_apriltags", "apriltag"]:
        spec = importlib.util.find_spec(m)
        if spec is None:
            print(f"Module {m}: NOT FOUND")
        else:
            try:
                mod = importlib.import_module(m)
                ver = getattr(mod, "__version__", None)
                path = getattr(mod, "__file__", None)
                print(f"Module {m}: FOUND version={ver} path={path}")
            except Exception as e:
                print(f"Module {m}: FOUND but import failed: {e!r}")

def load_gray(p: str) -> np.ndarray:
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Failed to load image: {p}")
    return g

def run_pupil(gray: np.ndarray) -> List[dict]:
    from pupil_apriltags import Detector  # type: ignore
    configs = [
        dict(families="tag36h11,tagStandard41h12,tag25h9,tag16h5", quad_decimate=1.0, quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25),
        dict(families="tag36h11", quad_decimate=1.0, quad_sigma=0.8, refine_edges=True, decode_sharpening=0.25),
        dict(families="tag36h11", quad_decimate=0.5, quad_sigma=0.8, refine_edges=True, decode_sharpening=0.5),
    ]
    for i, cfg in enumerate(configs):
        det = Detector(nthreads=1, **cfg)
        res = det.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
        print(f" pupil cfg{i}: {cfg} -> {len(res)} detections")
        if res:
            return [{"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)} for r in res]
    return []

def run_apr(gray: np.ndarray) -> List[dict]:
    import apriltag as apr  # type: ignore
    fams = ["tag36h11", "tagStandard41h12", "tag25h9", "tag16h5"]
    for fam in fams:
        for dec in (1.0, 0.5):
            for blur in (0.0, 0.8, 1.2):
                opts = apr.DetectorOptions(
                    families=fam,
                    border=1,
                    nthreads=1,
                    quad_decimate=dec,
                    quad_blur=blur,
                    refine_edges=True,
                    refine_decode=True,
                    refine_pose=False,
                    quad_contours=True,
                )
                det = apr.Detector(opts)
                res = det.detect(gray)
                print(f" apr fam={fam} dec={dec} blur={blur} -> {len(res)} detections")
                if res:
                    return [{"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)} for r in res]
    return []

def main():
    print_env()
    img_files = [f"datarow-00001-cam{i}.jpg" for i in (1, 2, 3)]
    for p in img_files:
        if not os.path.isfile(os.path.join(SCRIPT_DIR, p)):
            print(f"Image missing: {p}")
    print("Starting probes...\n")

    for cam in (1, 2, 3):
        path = os.path.join(SCRIPT_DIR, f"datarow-00001-cam{cam}.jpg")
        if not os.path.isfile(path):
            print(f"[cam{cam}] skip, not found: {path}")
            continue
        gray = load_gray(path)
        print(f"[cam{cam}] image {path} shape={gray.shape} min={int(gray.min())} max={int(gray.max())}")

        found = []

        # 1) Sweeper if present
        if sweeper is not None:
            try:
                res = sweeper(gray)
                print(f"[cam{cam}] sweeper -> {len(res)} detections")
                found = res
            except Exception as e:
                print(f"[cam{cam}] sweeper error: {e!r}")

        # 2) pupil_apriltags direct
        if not found:
            try:
                res = run_pupil(gray)  # run_pupil() [probe_apriltag.py](./probe_apriltag.py:33)
                print(f"[cam{cam}] pupil_apriltags -> {len(res)} detections")
                found = res
            except Exception as e:
                print(f"[cam{cam}] pupil_apriltags error: {e!r}")

        # 3) python-apriltag direct
        if not found:
            try:
                res = run_apr(gray)  # run_apr() [probe_apriltag.py](./probe_apriltag.py:44)
                print(f"[cam{cam}] python-apriltag -> {len(res)} detections")
                found = res
            except Exception as e:
                print(f"[cam{cam}] python-apriltag error: {e!r}")

        # Save visualization if any
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        vis = color.copy()
        for d in found:
            pts = d["corners"].astype(int).reshape(-1, 2)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.putText(vis, f"id={d['id']}", tuple(pts[0] + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = os.path.splitext(path)[0] + "_probe.jpg"
        cv2.imwrite(out, vis)
        print(f"[cam{cam}] wrote {out} with {len(found)} detections\n")

if __name__ == "__main__":
    main()  # [probe_apriltag.main()](./probe_apriltag.py:69)