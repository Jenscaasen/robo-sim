import numpy as np

def get_detector():
    """
    Return a callable detect(gray: np.ndarray) -> List[{'id': int, 'corners': (4,2) float32}]
    Tries pupil_apriltags first (fast), then AprilRobotics python-apriltag.
    Each backend sweeps through a few parameter sets to improve detection on small/rendered tags.
    """
    # Try pupil_apriltags
    try:
        from pupil_apriltags import Detector as PupilDetector  # type: ignore

        def detect_with_pupil(gray):
            families_list = ["tag36h11", "tag25h9", "tag16h5", "tagStandard41h12"]
            param_grid = [
                dict(quad_decimate=1.0, quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25),
                dict(quad_decimate=1.0, quad_sigma=0.8, refine_edges=True, decode_sharpening=0.25),
                dict(quad_decimate=0.5, quad_sigma=0.8, refine_edges=True, decode_sharpening=0.5),
            ]
            for fams in [",".join(families_list), "tag36h11"]:
                for p in param_grid:
                    det = PupilDetector(families=fams, nthreads=1, **p)
                    results = det.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
                    if results:
                        return [
                            {"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)}
                            for r in results
                        ]
            return []

        return detect_with_pupil
    except Exception:
        pass

    # Try AprilRobotics python-apriltag
    try:
        import apriltag as apr  # type: ignore

        def detect_with_apr(gray):
            families_list = ["tag36h11", "tag25h9", "tag16h5", "tagStandard41h12"]
            for fam in families_list:
                for quad_decimate in (1.0, 0.5):
                    for quad_blur in (0.0, 0.8, 1.2):
                        options = apr.DetectorOptions(
                            families=fam,
                            border=1,
                            nthreads=1,
                            quad_decimate=quad_decimate,
                            quad_blur=quad_blur,
                            refine_edges=True,
                            refine_decode=True,
                            refine_pose=False,
                            quad_contours=True,
                        )
                        detector = apr.Detector(options)
                        results = detector.detect(gray)
                        if results:
                            return [
                                {"id": int(r.tag_id), "corners": np.array(r.corners, dtype=np.float32)}
                                for r in results
                            ]
            return []

        return detect_with_apr
    except Exception:
        pass

    raise ImportError(
        "No AprilTag detector available. Install one of: 'pip install pupil-apriltags' or 'pip install apriltag'."
    )