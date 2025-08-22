import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU for this script
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # ensure XLA JIT is off

from tensorflow import keras
import numpy as np, json
from GetPixelsFromLiveCam import get_pixels_with_fallback, get_pixels_from_cameras
from api_client import send_joint_positions, build_base_url
# ──────────────────────────────────────────────────────────────
# 7.  Ready-to-use predictor
# ──────────────────────────────────────────────────────────────
def predict_angles(cam1_xy, cam2_xy, cam3_xy):
    """
    Parameters
    ----------
    cam1_xy, cam2_xy, cam3_xy : (x,y) pixel tuples from 640×480 images.

    Returns
    -------
    np.ndarray shape (5,)  →  angles in original [-1.5, +1.5] radians
    """
    sample = np.array([
        cam1_xy[0], cam1_xy[1],
        cam2_xy[0], cam2_xy[1],
        cam3_xy[0], cam3_xy[1]], dtype=np.float32)[None, :]   # shape (1,6)

    # same scaling as during training
    sample[:,0::2] /= 639.0
    sample[:,1::2] /= 479.0

    pred_scaled = model.predict(sample, verbose=0)[0]
    return pred_scaled * 1.5         # back to original units


model = keras.models.load_model('best_inverse_kinematics.keras')

def main():
    """Main function for run2.py"""
    print("🤖 ARM POSITION ESTIMATOR - TensorFlow Version")
    print("=" * 50)
    
    # Get pixel coordinates using the reusable module
    goal_pixels, current_angles = get_pixels_with_fallback()
    
    if goal_pixels is None or current_angles is None:
        print("❌ Failed to get pixel coordinates")
        return
    
    print(f"🎯 Target pixel coordinates: {goal_pixels}")
    print(f"🔧 Current joint angles: {current_angles}")
    
    # Convert to camera-specific format for the TensorFlow model
    cam1_xy = (goal_pixels[0], goal_pixels[1])
    cam2_xy = (goal_pixels[2], goal_pixels[3])
    cam3_xy = (goal_pixels[4], goal_pixels[5])
    
    # Predict angles using the TensorFlow model
    predicted_angles = predict_angles(cam1_xy, cam2_xy, cam3_xy)
    
    print("\n" + "=" * 50)
    print("📊 RESULTS - TensorFlow Model")
    print("=" * 50)
    print("Predicted joint angles (radians):")
    joints = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
    for name, val in zip(joints, predicted_angles):
        print(f"  {name:<15s}: {val:+.6f}")
    
    print(f"\nSend these five angles to your robot in this order:")
    print(f"[{', '.join(f'{x:.6f}' for x in predicted_angles)}]")
    
    # Ask user if they want to send the angles to the robot
    send_to_robot = input("🤖 Send these angles to the robot? (y/N): ").strip().lower()
    if send_to_robot in ['y', 'yes']:
        base_url = build_base_url("127.0.0.1", 5000)  # Default host and port
        success = send_joint_positions(base_url, predicted_angles)
        if success:
            print("✅ Angles sent to robot successfully!")
        else:
            print("❌ Failed to send angles to robot")
    else:
        print("⚠ Angles not sent to robot")

if __name__ == "__main__":
    main()