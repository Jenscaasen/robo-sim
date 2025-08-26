#!/usr/bin/env python3
# -----------------------------------------------------------
# Enhanced inverse_pixel_to_angles.py
#
#  (1) Connect to PyBullet backend API
#  (2) Get current joint positions from the robot
#  (3) Capture images from 3 cameras
#  (4) Detect red cube in the images
#  (5) Use ML model to estimate joint angles for target position
# -----------------------------------------------------------

import joblib
import numpy as np
import torch
import torch.nn as nn

from GetPixelsFromLiveCam import get_pixels_with_fallback
from api_client import send_joint_positions, build_base_url

# Configuration
API_HOST = "127.0.0.1"
API_PORT = 5000
BASE_URL = f"http://{API_HOST}:{API_PORT}"

MODEL_FILE = "arm_position_estimator_model.pkl"
SCALER_FILE = "scaler_params.pkl"

# -----------------------------------------------------------------
# 1. Load the trained scikit model and the scaling parameters
# -----------------------------------------------------------------
try:
    mlp = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_mean = torch.tensor(scaler["X_mean"], dtype=torch.float32)  # angles
    X_std = torch.tensor(scaler["X_std"], dtype=torch.float32)
    Y_mean = torch.tensor(scaler["Y_mean"], dtype=torch.float32)  # pixels
    Y_std = torch.tensor(scaler["Y_std"], dtype=torch.float32)
    
    # Build PyTorch network
    layers = []
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        in_dim, out_dim = w.shape
        lin = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w.T))
            lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
        if i < len(mlp.coefs_) - 1:
            layers.append(nn.Tanh())
    
    net = nn.Sequential(*layers)
    net.eval()
    print("✓ ML model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load ML model: {e}")
    print("  Manual input mode will be used instead")
    net = None

# -----------------------------------------------------------------
# 2. Backend API Communication Functions
# -----------------------------------------------------------------
# API calls moved to api_client.py

# Camera capture moved to api_client.py

# -----------------------------------------------------------------
# 3. Red Cube Detection Functions
# -----------------------------------------------------------------
# Red detection moved to vision/red_detector.py

# -----------------------------------------------------------------
# 4. ML Model Functions
# -----------------------------------------------------------------
def pixels_to_angles(goal_px, theta_start, lr=0.05, steps=20):
    """
    Optimize angles so that predicted pixels ≈ goal pixels.
    
    Args:
        goal_px: iterable with 6 floats (u1,v1,u2,v2,u3,v3)
        theta_start: iterable with 5 floats (initial guess, e.g. current pose)
        lr: learning rate
        steps: optimization steps
        
    Returns:
        numpy array (5,) with angles in radians
    """
    if net is None:
        print("✗ ML model not available")
        return None
    
    goal_px = torch.tensor(goal_px, dtype=torch.float32)
    theta = torch.tensor(theta_start, dtype=torch.float32, requires_grad=True)
    
    opt = torch.optim.SGD([theta], lr=lr)
    
    print(f"🧠 Running ML optimization ({steps} steps)...")
    initial_loss = None
    
    for step in range(steps):
        opt.zero_grad()
        theta_n = (theta - X_mean) / X_std
        pred_px_n = net(theta_n)
        pred_px = pred_px_n * Y_std + Y_mean
        loss = ((pred_px - goal_px)**2).sum()
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        opt.step()
        
        if step % 5 == 0 or step == steps - 1:
            print(f"  Step {step+1:2d}: loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    print(f"✓ Optimization complete. Loss improved by {improvement:.1f}%")
    
    return theta.detach().numpy()

# -----------------------------------------------------------------
# 5. Main Application Functions
# -----------------------------------------------------------------

def main():
    """Main application entry point."""
    print("=" * 60)
    print("🤖 ARM POSITION ESTIMATOR - Enhanced Version")
    print("=" * 60)
    print()
    print("This tool will:")
    print("  1. Connect to the PyBullet robot API")
    print("  2. Get current joint positions")
    print("  3. Capture images from 3 cameras")
    print("  4. Detect the red cube position")
    print("  5. Calculate required joint angles using ML")
    print()
    
    # Use the reusable module to get pixel coordinates
    goal, theta0 = get_pixels_with_fallback(API_HOST, API_PORT)
    
    # Run ML optimization if model is available
    if net is not None:
        print(f"\n🎯 Target pixel coordinates: {goal}")
        print(f"🔧 Starting joint angles: {[f'{x:+.4f}' for x in theta0]}")
        
        angles = pixels_to_angles(goal, theta0)
        
        if angles is not None:
            print("\n" + "=" * 50)
            print("📊 RESULTS")
            print("=" * 50)
            print("Predicted joint angles (radians):")
            joints = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
            for name, val in zip(joints, angles):
                print(f"  {name:<15s}: {val:+.6f}")
            
            print(f"\nSend these five angles to your robot in this order:")
            print(f"[{', '.join(f'{x:.6f}' for x in angles)}]")
            print()
            
            # Ask user if they want to send the angles to the robot
            send_to_robot = input("🤖 Send these angles to the robot? (y/N): ").strip().lower()
            if send_to_robot in ['y', 'yes']:
                base_url = build_base_url(API_HOST, API_PORT)
                success = send_joint_positions(base_url, angles)
                if success:
                    print("✅ Angles sent to robot successfully!")
                else:
                    print("❌ Failed to send angles to robot")
            else:
                print("⚠ Angles not sent to robot")
        else:
            print("✗ ML optimization failed")
    else:
        print("✗ ML model not available - cannot perform angle estimation")

if __name__ == "__main__":
    main()