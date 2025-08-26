import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

from RedDetector import RedPurpleDetector, Detection  # local helper


@dataclass
class ViewDetections:
    red: Detection
    purple: Detection
    size: Tuple[int, int]  # (width, height)


class ArmPositionEnv(gym.Env):
    """
    Gymnasium environment that controls a robot arm over HTTP to align the gripper
    above a red cube using multi-view visual feedback.

    Success criterion (configurable):
      - Front view: |x_purple - x_red| <= eps_front (pixels)
      - Side view:  |x_purple - x_red| <= eps_side  AND  y_purple + margin <= y_red
      - Top view (optional): Euclidean distance <= eps_top (pixels)
      All of the above must hold for success_frames consecutive steps.

    Observation (vector):
      - Joint positions (normalized to [-1, 1]) for joints 0..4  -> 5
      - Joint velocities (normalized by maxVelocity, clipped to [-1, 1]) -> 5
      - For each used camera (front, side, optional top):
          red (cx, cy in [0,1], or -1 if missing) -> 2
          purple (cx, cy in [0,1], or -1 if missing) -> 2
          detection flags: red_found, purple_found -> 2
        => per-view 6 features. With 2 views = 12; with 3 views = 18.
      Total: 10 + 6 * n_views features.

    Action:
      - Continuous, shape (5,), values in [-1, 1], scaled to small delta radians per step.
        Absolute joint targets are sent via POST /api/joints (clamped to limits).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        camera_front: int = 3,
        camera_side: int = 2,
        camera_top: Optional[int] = None,  # None disables top-based checks
        action_scale_rad: float = 0.03,
        success_eps_front_px: int = 8,
        success_eps_side_px: int = 8,
        success_eps_top_px: int = 12,
        height_margin_px: int = 8,
        success_frames: int = 5,
        max_steps: int = 300,
        fail_vision_patience: int = 10,
        request_timeout_s: float = 2.0,
        move_penalty: float = 0.02,
        weights: Tuple[float, float, float, float] = (1.0, 0.6, 0.6, 0.6),  # (top, front, side, vertical)
        randomize_reset: bool = True,
        random_reset_scale: float = 0.15,  # radians around center (clamped by limits)
        use_fast_endpoint: bool = True,
    ):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.cam_ids = {"front": camera_front, "side": camera_side, "top": camera_top}
        self.use_top = camera_top is not None
        self.timeout = float(request_timeout_s)

        self.action_scale = float(action_scale_rad)
        self.success_eps_front_px = int(success_eps_front_px)
        self.success_eps_side_px = int(success_eps_side_px)
        self.success_eps_top_px = int(success_eps_top_px)
        self.height_margin_px = int(height_margin_px)
        self.success_frames = int(success_frames)
        self.max_steps = int(max_steps)
        self.fail_vision_patience = int(fail_vision_patience)
        self.move_penalty = float(move_penalty)
        self.weights = weights
        self.randomize_reset = bool(randomize_reset)
        self.random_reset_scale = float(random_reset_scale)
        self.use_fast = bool(use_fast_endpoint)

        self.detector = RedPurpleDetector()

        # Fetch joint limits and meta once
        self.joint_ids = [0, 1, 2, 3, 4]
        self._refresh_joint_limits()

        # Observation and action spaces
        n_views = 2 + (1 if self.use_top else 0)
        obs_dim = 10 + 6 * n_views
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_ids),), dtype=np.float32)

        # Episode state
        self._step_count = 0
        self._success_streak = 0
        self._vision_fail_streak = 0

    # --------------- Gym API ---------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._success_streak = 0
        self._vision_fail_streak = 0

        # Reset robot (server expects GET for instant reset)
        r = requests.get(f"{self.base_url}/api/reset/instant", timeout=self.timeout)
        r.raise_for_status()

        # Optional: slight randomization of initial joint pose
        self._refresh_joint_limits()  # ensure limits are cached
        joints = self._get_joints()
        curr = np.array([joints[str(i)]["current"] for i in self.joint_ids], dtype=np.float32)
        if self.randomize_reset:
            rng = np.random.default_rng(seed)
            deltas = rng.uniform(-self.random_reset_scale, self.random_reset_scale, size=len(self.joint_ids))
            targets = self._clamp_to_limits(curr + deltas)
            self._set_joints_abs(targets)

        obs, _views = self._build_observation()
        info = {"views": _views}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Convert action in [-1,1] to small delta radians
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        deltas = action * self.action_scale

        # Read current joints and send absolute positions
        joints = self._get_joints()
        curr = np.array([joints[str(i)]["current"] for i in self.joint_ids], dtype=np.float32)
        targets = self._clamp_to_limits(curr + deltas)
        self._set_joints_abs(targets)

        # Observe camera + joints, compute reward and done
        obs, views = self._build_observation(joints_cache=joints)
        reward, success_now, vision_ok = self._compute_reward_and_success(views, deltas)

        # Success streaks and failures
        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        if vision_ok:
            self._vision_fail_streak = 0
        else:
            self._vision_fail_streak += 1

        terminated = self._success_streak >= self.success_frames
        truncated = self._step_count >= self.max_steps or self._vision_fail_streak >= self.fail_vision_patience

        info = {
            "success_streak": self._success_streak,
            "vision_fail_streak": self._vision_fail_streak,
            "views": views,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------------- Observation / Reward ---------------

    def _build_observation(self, joints_cache: Optional[dict] = None):
        """
        Returns (obs_vector, views_dict)
        views_dict: { view_name: ViewDetections }
        """
        # Joints
        joints = joints_cache or self._get_joints()
        pos = []
        vel = []
        for i in self.joint_ids:
            ji = joints[str(i)]
            pos.append(self._normalize_pos(i, float(ji["current"])))
            max_v = max(float(ji.get("maxVelocity", 1.0)), 1e-6)
            vel.append(np.clip(float(ji.get("velocity", 0.0)) / max_v, -1.0, 1.0))
        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)

        # Cameras
        views: Dict[str, ViewDetections] = {}
        for name, cam_id in self.cam_ids.items():
            if cam_id is None:
                continue
            img = self._get_camera(cam_id)
            h, w = (img.shape[0], img.shape[1]) if img is not None else (0, 0)
            det = {"red": Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0)),
                   "purple": Detection(False, -1.0, -1.0, 0.0, (0, 0, 0, 0))}
            if img is not None and img.size > 0:
                d = self.detector.detect_colors(img)
                det = d
            views[name] = ViewDetections(red=det["red"], purple=det["purple"], size=(w, h))

        # Build feature vector
        feats: List[float] = []
        feats.extend(pos.tolist())
        feats.extend(vel.tolist())
        for key in ("front", "side", "top"):
            if key not in views or (key == "top" and not self.use_top):
                continue
            vw = views[key]
            w, h = vw.size
            # Normalize to [0,1], -1 if not found
            for obj in (vw.red, vw.purple):
                if obj.found and w > 0 and h > 0:
                    feats.append(float(obj.cx) / float(w))
                    feats.append(float(obj.cy) / float(h))
                else:
                    feats.extend([-1.0, -1.0])
            feats.append(1.0 if vw.red.found else 0.0)
            feats.append(1.0 if vw.purple.found else 0.0)

        obs = np.asarray(feats, dtype=np.float32)
        return obs, views

    def _compute_reward_and_success(self, views: Dict[str, ViewDetections], deltas: np.ndarray):
        """
        Returns (reward, success_now, vision_ok)
        """
        w_top, w_front, w_side, w_vert = self.weights
        reward = 0.0
        success_front = False
        success_side = False
        success_top = not self.use_top  # if top is disabled, treat as satisfied
        vision_ok = True

        # Front view: align x
        if "front" in views:
            vf = views["front"]
            if vf.red.found and vf.purple.found and vf.size[0] > 0:
                dx = abs(vf.purple.cx - vf.red.cx)
                dx_norm = dx / float(vf.size[0])
                reward -= w_front * dx_norm
                success_front = dx <= self.success_eps_front_px
            else:
                vision_ok = False
                reward -= 0.05

        # Side view: align x, keep gripper above cube (y smaller)
        if "side" in views:
            vs = views["side"]
            if vs.red.found and vs.purple.found and vs.size[0] > 0 and vs.size[1] > 0:
                dx = abs(vs.purple.cx - vs.red.cx)
                dx_norm = dx / float(vs.size[0])
                reward -= w_side * dx_norm

                # vertical margin: y increases downward in images
                viol_px = max(0.0, (vs.purple.cy) - (vs.red.cy - self.height_margin_px))
                viol_norm = viol_px / float(vs.size[1])
                reward -= w_vert * viol_norm

                success_side = (dx <= self.success_eps_side_px) and (viol_px <= 0.5)  # allow tiny slack
            else:
                vision_ok = False
                reward -= 0.05

        # Top view (optional): 2D distance
        if self.use_top and "top" in views:
            vt = views["top"]
            if vt.red.found and vt.purple.found and vt.size[0] > 0 and vt.size[1] > 0:
                dx = (vt.purple.cx - vt.red.cx)
                dy = (vt.purple.cy - vt.red.cy)
                dist = (dx * dx + dy * dy) ** 0.5
                diag = (vt.size[0] ** 2 + vt.size[1] ** 2) ** 0.5
                reward -= w_top * (dist / float(diag))
                success_top = dist <= self.success_eps_top_px
            else:
                vision_ok = False
                reward -= 0.05

        # Motion penalty
        move_cost = self.move_penalty * float(np.mean(np.abs(deltas)))
        reward -= move_cost

        success_now = success_front and success_side and success_top
        if success_now:
            reward += 1.0  # one-time bonus when all conditions satisfied

        return reward, success_now, vision_ok

    # --------------- HTTP helpers ---------------

    def _get_joints(self) -> dict:
        r = requests.get(f"{self.base_url}/api/joints", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _set_joints_abs(self, targets: np.ndarray):
        payload = [{"id": int(i), "pos": float(p)} for i, p in zip(self.joint_ids, targets.tolist())]
        # Try fast endpoint first if enabled; fall back to standard endpoint on 404/405 or errors
        if getattr(self, "use_fast", False):
            try:
                self._post_json("/api/joints/fast", payload)
                return
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (404, 405):
                    self.use_fast = False
                else:
                    raise
            except Exception:
                self.use_fast = False
        self._post_json("/api/joints", payload)

    def _get_camera(self, cam_id: int) -> Optional[np.ndarray]:
        try:
            r = requests.get(
                f"{self.base_url}/api/camera/{cam_id}",
                headers={"Accept": "image/jpeg"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _post_json(self, path: str, payload):
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if payload is None:
            r = requests.post(url, timeout=self.timeout, headers=headers)
        else:
            r = requests.post(url, json=payload, timeout=self.timeout, headers=headers)
        r.raise_for_status()

    # --------------- Joint limits / normalization ---------------

    def _refresh_joint_limits(self):
        joints = self._get_joints()
        lowers = []
        uppers = []
        max_vel = []
        for i in self.joint_ids:
            ji = joints[str(i)]
            lowers.append(float(ji["lower"]))
            uppers.append(float(ji["upper"]))
            max_vel.append(max(float(ji.get("maxVelocity", 1.0)), 1e-6))
        self._lower = np.asarray(lowers, dtype=np.float32)
        self._upper = np.asarray(uppers, dtype=np.float32)
        self._range = np.maximum(self._upper - self._lower, 1e-6)
        self._center = (self._upper + self._lower) * 0.5
        self._max_vel = np.asarray(max_vel, dtype=np.float32)

    def _normalize_pos(self, idx: int, pos: float) -> float:
        # map to [-1, 1] using joint limits
        return float(np.clip(2.0 * (pos - self._center[idx]) / self._range[idx], -1.0, 1.0))

    def _clamp_to_limits(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, self._lower, self._upper).astype(np.float32)


# Convenience factory
def make_env(**kwargs) -> gym.Env:
    return ArmPositionEnv(**kwargs)