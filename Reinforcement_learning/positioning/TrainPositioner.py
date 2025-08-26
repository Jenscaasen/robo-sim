import os
import argparse
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

# Import the environment as a package module so relative imports inside it work
from ArmPositionEnv import make_env


def build_env_factory(args: argparse.Namespace) -> Callable[[], gym.Env]:
    def _f():
        # Enable top camera by default for robust success test; can disable via --no-top
        cam_top = None if args.no_top else args.cam_top
        env = make_env(
            host=args.host,
            port=args.port,
            camera_front=args.cam_front,
            camera_side=args.cam_side,
            camera_top=cam_top,
            action_scale_rad=args.action_scale,
            success_eps_front_px=args.eps_front,
            success_eps_side_px=args.eps_side,
            success_eps_top_px=args.eps_top,
            height_margin_px=args.h_margin,
            success_frames=args.success_frames,
            max_steps=args.max_steps,
            move_penalty=args.move_penalty,
            fail_vision_patience=args.vision_patience,
            randomize_reset=not args.no_random_reset,
            use_fast_endpoint=not args.no_fast,
        )
        return env
    return _f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO to position gripper above red cube (multi-view).")
    # HTTP endpoints
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)

    # Cameras
    p.add_argument("--cam-front", type=int, default=3, dest="cam_front", help="Front/angled camera id")
    p.add_argument("--cam-side", type=int, default=2, dest="cam_side", help="Side camera id")
    p.add_argument("--cam-top", type=int, default=1, dest="cam_top", help="Top-down camera id")
    p.add_argument("--no-top", action="store_true", help="Disable using the top camera in success criterion")
    p.add_argument("--no-fast", action="store_true", help="Disable /api/joints/fast and use /api/joints instead")

    # RL hyperparameters and env config
    p.add_argument("--total-timesteps", type=int, default=200_000, dest="total_timesteps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (HTTP load grows with this)")
    p.add_argument("--action-scale", type=float, default=0.05, dest="action_scale")
    p.add_argument("--eps-front", type=int, default=8, dest="eps_front")
    p.add_argument("--eps-side", type=int, default=8, dest="eps_side")
    p.add_argument("--eps-top", type=int, default=12, dest="eps_top")
    p.add_argument("--h-margin", type=int, default=8, dest="h_margin")
    p.add_argument("--success-frames", type=int, default=5, dest="success_frames")
    p.add_argument("--max-steps", type=int, default=300, dest="max_steps")
    p.add_argument("--vision-patience", type=int, default=10, dest="vision_patience")
    p.add_argument("--move-penalty", type=float, default=0.02, dest="move_penalty")
    p.add_argument("--no-random-reset", action="store_true")

    # Logging / checkpoints
    p.add_argument("--log-dir", type=str, default="runs/positioner")
    p.add_argument("--save-freq", type=int, default=50_000, help="Timesteps between checkpoints")
    p.add_argument("--tensorboard", action="store_true", help="Enable SB3 TensorBoard logger")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # Vectorized env (keep n-envs modest to avoid overloading HTTP sim)
    factory = build_env_factory(args)
    if args.n_envs == 1:
        venv = DummyVecEnv([factory])
    else:
        venv = DummyVecEnv([factory for _ in range(args.n_envs)])
    venv = VecMonitor(venv)  # episode stats

    # Logger
    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"] if args.tensorboard else ["stdout", "csv"])

    # PPO configuration: conservative defaults
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        seed=args.seed,
        tensorboard_log=args.log_dir if args.tensorboard else None,
    )
    model.set_logger(new_logger)

    # Checkpoint callback
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq // max(args.n_envs, 1),
                                             save_path=ckpt_dir,
                                             name_prefix="ppo_positioner")

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(args.log_dir, "ppo_positioner_final"))


if __name__ == "__main__":
    main()