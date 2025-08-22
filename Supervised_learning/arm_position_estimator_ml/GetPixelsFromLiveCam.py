#!/usr/bin/env python3
# -----------------------------------------------------------
# GetPixelsFromLiveCam.py
#
# Reusable module for getting pixel coordinates from cameras
# Supports both automatic (API + vision) and manual input modes
# -----------------------------------------------------------

import os
from typing import List, Optional, Tuple

from api_client import build_base_url, get_current_joint_positions, capture_all_cameras
from vision.red_detector import detect_red_cube_in_all_cameras


def get_pixels_automatic_mode(host: str = "127.0.0.1", port: int = 5000, 
                             debug: bool = True, debug_dir: str = ".") -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Get pixel coordinates automatically using API and computer vision.
    
    Args:
        host: API host address
        port: API port
        debug: Whether to save debug overlays
        debug_dir: Directory to save debug images
        
    Returns:
        Tuple of (goal_pixels, current_joint_angles) or (None, None) on failure
    """
    print("\n=== Automatic Mode ===")
    print("Connecting to robot and detecting red cube...\n")
    
    base_url = build_base_url(host, port)
    
    # Step 1: Get current joint positions
    print("📡 Getting current joint positions...")
    current_positions, joints_data = get_current_joint_positions(base_url)
    if current_positions is None:
        return None, None
    
    # Step 2: Capture camera images
    print("\n📷 Capturing camera images...")
    images = capture_all_cameras(base_url)
    if images is None:
        print("✗ Failed to capture all camera images")
        return None, None
    
    # Step 3: Detect red cube
    print("\n🎯 Detecting red cube...")
    cube_positions = detect_red_cube_in_all_cameras(images, debug=debug, debug_dir=debug_dir)
    if cube_positions is None:
        print("✗ Failed to detect red cube in all cameras")
        return None, None
    
    # Convert positions to goal format (flatten list of tuples)
    goal = []
    for x, y in cube_positions:
        goal.extend([x, y])
    
    return goal, current_positions


def get_pixels_manual_mode() -> Tuple[List[float], List[float]]:
    """
    Get pixel coordinates through manual user input.
    
    Returns:
        Tuple of (goal_pixels, current_joint_angles)
    """
    def ask_float(prompt):
        """Helper function to get float input from user."""
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("  Please enter a valid number")
    
    print("\n=== Manual Input Mode ===")
    print("Enter target pixel coordinates and starting joint angles manually.\n")
    
    # Get target pixel coordinates
    goal = []
    for cam in (1, 2, 3):
        print(f"Camera {cam} target position:")
        u = ask_float(f"  u (x): ")
        v = ask_float(f"  v (y): ")
        goal.extend([u, v])
    
    # Get starting joint angles
    print("\nStarting joint angles (radians). Press Enter for current/zero:")
    joints = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
    theta0 = []
    for j in joints:
        s = input(f"  {j}: ")
        theta0.append(float(s) if s.strip() else 0.0)
    
    return goal, theta0


def get_pixels_with_fallback(host: str = "127.0.0.1", port: int = 5000, 
                            debug: bool = True, debug_dir: str = ".") -> Tuple[List[float], List[float]]:
    """
    Get pixel coordinates with automatic fallback to manual mode.
    
    Args:
        host: API host address
        port: API port
        debug: Whether to save debug overlays
        debug_dir: Directory to save debug images
        
    Returns:
        Tuple of (goal_pixels, current_joint_angles)
    """
    # Try automatic mode first
    goal, theta0 = get_pixels_automatic_mode(host, port, debug, debug_dir)
    
    # Fall back to manual mode if automatic fails
    if goal is None or theta0 is None:
        print("\n⚠ Automatic mode failed. Switching to manual input mode...")
        goal, theta0 = get_pixels_manual_mode()
    
    return goal, theta0


def get_pixels_from_cameras(cam1_xy: Tuple[float, float], 
                           cam2_xy: Tuple[float, float], 
                           cam3_xy: Tuple[float, float]) -> List[float]:
    """
    Convert camera pixel coordinates to the format expected by ML models.
    
    Args:
        cam1_xy, cam2_xy, cam3_xy: (x,y) pixel tuples from cameras
        
    Returns:
        List of 6 pixel coordinates in format [u1, v1, u2, v2, u3, v3]
    """
    return [
        cam1_xy[0], cam1_xy[1],
        cam2_xy[0], cam2_xy[1], 
        cam3_xy[0], cam3_xy[1]
    ]


if __name__ == "__main__":
    # Test the module
    print("Testing GetPixelsFromLiveCam module...")
    pixels, angles = get_pixels_with_fallback()
    
    if pixels and angles:
        print(f"\n✅ Got pixels: {pixels}")
        print(f"✅ Got angles: {angles}")
    else:
        print("\n❌ Failed to get pixel coordinates")