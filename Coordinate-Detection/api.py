#!/usr/bin/env python3
# Lightweight HTTP client for the robot and cameras
# Uses the HTTP contract described in Coordinate-Detection/httpinterface.md

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional, Iterable

import requests
import numpy as np
import cv2


class RobotHttp:
    """
    Minimal client around the HTTP API.

    Endpoints used:
      - GET /api/joints (read joint states/limits)
      - POST /api/joints (absolute targets)
      - (optional) POST /api/joints/fast (if available; we auto-fallback)
      - GET /api/reset/instant
      - GET /api/camera/{id}
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._use_fast = True  # try fast endpoint first and fall back automatically

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -------------------------------
    # Joint state and limits
    # -------------------------------
    def get_joints(self) -> Dict:
        """Return raw JSON dict for all joints."""
        r = requests.get(f"{self.base_url}/api/joints", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_joint_limits(self) -> Dict[int, Tuple[float, float]]:
        """
        Parse limits for joints with 0..N integer indices.
        Returns: {joint_id: (lower, upper)}
        """
        data = self.get_joints()
        lims: Dict[int, Tuple[float, float]] = {}
        for k, v in data.items():
            try:
                jid = int(k)
            except Exception:
                continue
            lower = float(v.get("lower", -3.14159265))
            upper = float(v.get("upper", 3.14159265))
            lims[jid] = (lower, upper)
        return lims

    def get_current_positions(self, ids: Iterable[int] = range(5)) -> List[float]:
        """Return current positions for given joint ids (default 0..4)."""
        data = self.get_joints()
        out: List[float] = []
        for i in ids:
            j = data.get(str(i), {})
            out.append(float(j.get("current", 0.0)))
        return out

    # -------------------------------
    # Command joints
    # -------------------------------
    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def clamp_targets(self, targets: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Clamp list of {id,pos} against live limits."""
        lims = self.get_joint_limits()
        clamped: List[Dict[str, float]] = []
        for t in targets:
            jid = int(t["id"])
            pos = float(t["pos"])
            if jid in lims:
                lo, hi = lims[jid]
                pos = self._clamp(pos, lo, hi)
            clamped.append({"id": jid, "pos": pos})
        return clamped

    def set_joints(self, targets: List[Dict[str, float]], settle_s: float = 0.3) -> None:
        """
        Send absolute targets in the canonical payload:
          [{"id": 0, "pos": 0.1}, ...]
        We try /api/joints/fast first; if not supported we fall back to /api/joints.
        """
        payload = self.clamp_targets(targets)
        url_fast = f"{self.base_url}/api/joints/fast"
        url_std = f"{self.base_url}/api/joints"

        try:
            if self._use_fast:
                r = requests.post(url_fast, json=payload, timeout=self.timeout_s)
                if r.status_code in (404, 405):
                    self._use_fast = False  # disable fast, retry std
                else:
                    r.raise_for_status()
            if not self._use_fast:
                r = requests.post(url_std, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
        finally:
            if settle_s > 0:
                time.sleep(settle_s)

    # -------------------------------
    # Reset
    # -------------------------------
    def reset_instant(self) -> None:
        r = requests.get(f"{self.base_url}/api/reset/instant", timeout=self.timeout_s)
        r.raise_for_status()
        time.sleep(0.05)

    # -------------------------------
    # Cameras
    # -------------------------------
    def get_camera_image(self, cam_id: int) -> Optional[np.ndarray]:
        """
        Return OpenCV BGR image for the camera or None on decode error.
        """
        r = requests.get(f"{self.base_url}/api/camera/{cam_id}", headers={"Accept": "image/jpeg"}, timeout=10.0)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Camera {cam_id}: JPEG decode failed")
        return img

    def get_all_cameras(self, ids: Tuple[int, int, int] = (1, 2, 3)) -> Optional[List[np.ndarray]]:
        imgs: List[np.ndarray] = []
        for cid in ids:
            im = self.get_camera_image(cid)
            if im is None:
                return None
            imgs.append(im)
        return imgs


# -------------------------------
# Small helpers
# -------------------------------
def build_targets_from_list(positions: List[float], start_id: int = 0) -> List[Dict[str, float]]:
    """
    Convert [p0,p1,...] into [{"id":start_id+i, "pos":pi}, ...]
    """
    return [{"id": start_id + i, "pos": float(p)} for i, p in enumerate(positions)]


if __name__ == "__main__":
    # Quick smoke test (requires the simulator running)
    client = RobotHttp()
    try:
        print("Joint snapshot:", {k: (v.get("name"), v.get("current")) for k, v in client.get_joints().items()})
        img = client.get_camera_image(1)
        if img is not None:
            print("Camera 1 image:", img.shape)
    except Exception as e:
        print("HTTP test failed:", e)