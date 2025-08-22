#!/usr/bin/env python3
"""
Sticky Cube System Demonstration

This script demonstrates the sticky cube functionality by automatically moving
the robot arm to pick up cubes and place them in containers.
"""

import sys
import os
import time
import math

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    import pybullet as p
    import pybullet_data
except ImportError as e:
    print("PyBullet is not installed. Please install it first.")
    sys.exit(1)

from world_helpers import add_wall, spawn_cubes_on_floor, spawn_containers
from sticky_cube_system import StickyCubeSystem
from PybulletViewer import load_robot, setup_world, connect, get_controllable_joints


def move_robot_to_position(robot_id, joints, target_positions, duration=2.0):
    """
    Smoothly move robot to target joint positions over specified duration.
    
    Args:
        robot_id: PyBullet robot ID
        joints: List of joint information tuples
        target_positions: List of target joint angles (radians)
        duration: Time to complete the movement (seconds)
    """
    if len(target_positions) != len(joints):
        print(f"Error: Expected {len(joints)} positions, got {len(target_positions)}")
        return
    
    # Get current positions
    current_positions = []
    for joint_idx, _, _, _, _, _, _ in joints:
        joint_state = p.getJointState(robot_id, joint_idx)
        current_positions.append(joint_state[0])
    
    # Interpolate between current and target positions
    steps = int(duration * 240)  # 240 Hz simulation
    for step in range(steps):
        t = step / steps  # 0 to 1
        
        for i, (joint_idx, _, _, _, _, max_force, max_vel) in enumerate(joints):
            # Linear interpolation
            current_pos = current_positions[i]
            target_pos = target_positions[i]
            interpolated_pos = current_pos + t * (target_pos - current_pos)
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=interpolated_pos,
                force=max(1.0, max_force if max_force and not math.isinf(max_force) else 50.0),
                maxVelocity=max(0.1, max_vel if max_vel and not math.isinf(max_vel) else 2.0),
            )
        
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def get_end_effector_world_position(robot_id, end_effector_link_index):
    """Get the world position of the end effector"""
    if end_effector_link_index == -1:
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        return pos
    else:
        link_state = p.getLinkState(robot_id, end_effector_link_index)
        return link_state[0]


def demo_sticky_cubes():
    """Run the sticky cube demonstration"""
    print("🤖 Starting Sticky Cube Demonstration")
    print("=" * 50)
    
    # Connect to PyBullet
    cid = connect(gui=True)
    
    try:
        # Setup world
        setup_world(gravity=-9.81, add_plane=True)
        
        # Wall-mounted robot setup
        add_wall(0.0)
        base_pos = [0.0, 0.0, 0.75]
        base_orn = p.getQuaternionFromEuler([0, math.pi / 2.0, 0])
        
        # Load robot
        urdf_path = os.path.join("urdf", "five_dof_arm.urdf")
        robot_id = load_robot(urdf_path, fixed_base=True, base_pos=base_pos, base_orn=base_orn)
        
        # Get joints
        joints = get_controllable_joints(robot_id)
        print(f"Robot loaded with {len(joints)} controllable joints")
        
        # Find end effector link
        end_effector_link_index = len(joints) - 1  # Last joint link
        print(f"End effector link index: {end_effector_link_index}")
        
        # Initialize sticky cube system
        sticky_system = StickyCubeSystem(robot_id, end_effector_link_index)
        
        # Spawn cubes
        cube_ids = spawn_cubes_on_floor(3, x=0.5, y0=0.0, z=0.05, spacing=0.15)
        for cube_id in cube_ids:
            sticky_system.register_cube(cube_id)
        print(f"Spawned and registered {len(cube_ids)} cubes")
        
        # Spawn containers
        container_info = spawn_containers(2, base_x=1.2, base_y=0.0, spacing=0.4)
        for container_id, container_name in container_info:
            container_size = (0.2, 0.2, 0.15)
            container_index = container_info.index((container_id, container_name))
            y_pos = 0.0 + (container_index - (len(container_info) - 1) * 0.5) * 0.4
            position = (1.2, y_pos, container_size[2]/2)
            sticky_system.register_container(container_id, position, container_size, container_name)
        print(f"Spawned and registered {len(container_info)} containers")
        
        # Wait for user to see initial setup
        print("\n📋 Initial Setup Complete!")
        print("- Robot arm mounted on wall")
        print("- 3 colored cubes on the floor")
        print("- 2 containers ready to receive cubes")
        print("\nPress Enter to start the demonstration...")
        input()
        
        # Demonstration sequence
        print("\n🎬 Starting Demonstration Sequence")
        
        # Define some useful joint positions
        # Joint order: [shoulder_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll]
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0]  # All joints at zero
        reach_cube_positions = [
            [0.3, -0.5, 0.8, -0.3, -0.8],   # Reach for first cube (gripper active)
            [0.0, -0.5, 0.8, -0.3, -0.8],   # Reach for second cube (gripper active)
            [-0.3, -0.5, 0.8, -0.3, -0.8],  # Reach for third cube (gripper active)
        ]
        container_positions = [
            [0.8, -0.3, 0.5, -0.2, 0.8],   # Move to first container (gripper release)
            [0.8, 0.3, 0.5, -0.2, 0.8],    # Move to second container (gripper release)
        ]
        
        # Move to home position
        print("🏠 Moving to home position...")
        move_robot_to_position(robot_id, joints, home_position, duration=2.0)
        
        # Update sticky system for a few steps to ensure everything is initialized
        for _ in range(10):
            sticky_system.update()
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        
        # Demonstrate picking up and placing cubes
        for i, cube_id in enumerate(cube_ids[:2]):  # Pick up first 2 cubes
            print(f"\n📦 Demonstration {i+1}: Picking up cube {cube_id}")
            
            # Move to cube with gripper active (wrist roll < -0.5)
            print(f"   → Moving to cube {cube_id} with gripper active...")
            move_robot_to_position(robot_id, joints, reach_cube_positions[i], duration=3.0)
            
            # Give time for attachment to occur
            print(f"   → Waiting for cube attachment...")
            attachment_attempts = 0
            max_attempts = 50
            while sticky_system.attached_cube_id != cube_id and attachment_attempts < max_attempts:
                sticky_system.update()
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
                attachment_attempts += 1
            
            if sticky_system.attached_cube_id == cube_id:
                print(f"   ✅ Cube {cube_id} attached via gripper control!")
                
                # Move to container with gripper release (wrist roll > 0.5)
                container_idx = i % len(container_positions)
                print(f"   → Moving to container {container_idx + 1} and releasing gripper...")
                move_robot_to_position(robot_id, joints, container_positions[container_idx], duration=3.0)
                
                # Give time for release to occur
                print(f"   → Waiting for cube release...")
                release_attempts = 0
                max_release_attempts = 50
                while sticky_system.attached_cube_id == cube_id and release_attempts < max_release_attempts:
                    sticky_system.update()
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)
                    release_attempts += 1
                
                if sticky_system.attached_cube_id != cube_id:
                    print(f"   ✅ Cube {cube_id} released via gripper control!")
                else:
                    print(f"   ⚠️ Cube {cube_id} still attached (check gripper angle or container position)")
            else:
                print(f"   ⚠️ Failed to attach cube {cube_id} (check distance and gripper angle)")
            
            # Brief pause between demonstrations
            time.sleep(1.0)
        
        # Return to home position
        print("\n🏠 Returning to home position...")
        move_robot_to_position(robot_id, joints, home_position, duration=2.0)
        
        print("\n🎉 Demonstration Complete!")
        print("\nHow the Sticky Cube System Works:")
        print("1. 🎯 Distance Detection: Monitors distance between end effector and cubes")
        print("2. 🔗 Auto Attachment: Creates physics constraint when within 5cm threshold")
        print("3. 🚚 Transport: Cube follows end effector movements via rigid constraint")
        print("4. 📦 Auto Release: Removes constraint when cube enters container area")
        print("\nPress Enter to exit...")
        input()
        
    finally:
        sticky_system.cleanup()
        p.disconnect()


if __name__ == "__main__":
    demo_sticky_cubes()