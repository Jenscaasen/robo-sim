import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests, cv2, time
from functools import lru_cache

# ---------------------------------------------------------------------------
# Helper functions you said already exist
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def GetFrameWithHandAndTarget(jpg_bytes: bytes, size=64) -> np.ndarray:
    """
    Returns a size×size gray image that contains the gripper and the target.
    Dummy implementation: just resize the whole frame.
    Replace by your real cropping logic.
    """
    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def MoveHandTowardsTarget(base_url: str):
    """
    Your classic controller that does the coarse motion.
    """
    # --- demo: put arm in a predefined pose --------------------------------
    pose = [
        {"id": 0, "pos": 0.0},
        {"id": 1, "pos": 0.6},
        {"id": 2, "pos": 0.8},
        {"id": 3, "pos": 0.0},
        {"id": 4, "pos": 0.0},
        {"id": 5, "pos": 0.035},   # open gripper
        {"id": 6, "pos": 0.035}
    ]
    requests.post(f"{base_url}/api/joints/fast", json=pose)
    time.sleep(0.25)   # let the sim settle


# ---------------------------------------------------------------------------
# Gym environment
# ---------------------------------------------------------------------------
class FineGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 host="localhost", port=5000,
                 img_size=64,
                 torque_threshold=-20.0):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.img_size = img_size
        self.torque_threshold = torque_threshold

        # ----- ACTION SPACE : ABSOLUTE ANGLES (rad) -------------------------
        low  = np.array([-1.6, -1.6, -1.6, -1.6, -1.6,  0.00, 0.00],
                        dtype=np.float32)
        high = np.array([ 1.6,  1.6,  1.6,  1.6,  1.6,  0.04, 0.04],
                        dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # ----- OBSERVATION SPACE -------------------------------------------
        # camera : img_size×img_size gray  -> flattened
        cam_len = img_size * img_size
        # 7 joint angles + 2 finger torques
        low_obs  = np.concatenate([np.zeros(cam_len,          np.uint8),
                                   -1.6*np.ones(7,            np.float32),
                                   -40.0*np.ones(2,           np.float32)])
        high_obs = np.concatenate([255*np.ones(cam_len,       np.uint8),
                                   1.6*np.ones(7,             np.float32),
                                   40.0*np.ones(2,            np.float32)])
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        # internal
        self._cached_last_camera = None

    # -----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 1. hard reset the simulation
        requests.get(f"{self.base_url}/api/reset/instant")

        # 2. coarse motion controller brings wrist near object
        MoveHandTowardsTarget(self.base_url)

        obs = self._get_obs()
        self.step_counter = 0
        return obs, {}

    # -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        # clip for safety
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # send the command
        payload = [{"id": i, "pos": float(p)} for i, p in enumerate(action)]
        requests.post(f"{self.base_url}/api/joints/fast", json=payload)

        # wait one sim step (your HTTP server probably advances automatically)
        time.sleep(1/60)

        obs   = self._get_obs()
        rew, done, info = self._compute_reward_done()

        self.step_counter += 1
        if self.step_counter >= 150:   # time-out
            done = True

        return obs, rew, done, False, info

    # -----------------------------------------------------------------------
    def _get_obs(self):
        # ---- 1. CAMERA CROP ------------------------------------------------
        jpg  = requests.get(f"{self.base_url}/api/camera/1").content
        crop = GetFrameWithHandAndTarget(jpg, self.img_size)   # size×size
        flat = crop.flatten().astype(np.float32)

        # ---- 2. joint angles & torques -------------------------------------
        js   = requests.get(f"{self.base_url}/api/joints").json()
        angles  = np.array([js[str(i)]["current"] for i in range(7)],
                           dtype=np.float32)
        torques = np.array([js["5"]["applied_torque"],
                            js["6"]["applied_torque"]], dtype=np.float32)

        obs = np.concatenate([flat, angles, torques]).astype(np.float32)
        return obs

    # -----------------------------------------------------------------------
    def _compute_reward_done(self):
        js   = requests.get(f"{self.base_url}/api/joints").json()
        # finger torques (negative when pressing)
        tq = min(js["5"]["applied_torque"], js["6"]["applied_torque"])

        # shaping: small penalty each step
        reward = -0.01

        # bonus when gripper starts touching object
        if tq < self.torque_threshold:
            reward += 0.5

        # success when finger torque + gripper lifted 3 cm
        success = False
        if tq < self.torque_threshold:
            # read wrist height from a dedicated joint, or approximate
            # here we just mark success as soon as finger load reached threshold
            success = True

        done = bool(success)
        if success:
            reward += 5.0

        return reward, done, {"torque": tq}

    # -----------------------------------------------------------------------
    def render(self):
        if self._cached_last_camera is not None:
            cv2.imshow("crop", self._cached_last_camera)
            cv2.waitKey(1)

# ---------------------------------------------------------------------------
# MAIN: Train with SAC
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

    # ------------------------------------------------------------
    # run 8 docker containers → 8 parallel envs (ports 5000-5007)
    def make_env(rank):
        def _thunk():
            return FineGraspEnv(port=5000 + rank)
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(8)])
    env = VecTransposeImage(env)   # (C,H,W) order for SB3 when using images

    # ------------------------------------------------------------
    model = SAC(
        policy="CnnPolicy",        # has a default small CNN
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        verbose=1,
        tensorboard_log="./tb/"
    )

    model.learn(total_timesteps=3_000_000)
    model.save("fine_grasp_sac_abs")