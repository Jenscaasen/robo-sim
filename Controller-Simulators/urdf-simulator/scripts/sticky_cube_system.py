#!/usr/bin/env python3
"""
Sticky Cube System for PyBullet Robot Simulation

This module implements a physics-based system where cubes can "stick" to the robot's
end effector when touched, and be released when placed into containers.
"""

import pybullet as p
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class CubeState:
    """Represents the state of a cube in the simulation"""
    cube_id: int
    is_attached: bool = False
    constraint_id: Optional[int] = None
    original_position: Optional[Tuple[float, float, float]] = None


@dataclass
class Container:
    """Represents a container that can hold cubes"""
    container_id: int
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    name: str


class StickyCubeSystem:
    """
    Manages the sticky cube interaction system for a robotic arm simulation.
    
    Features:
    - Uses wrist roll joint as gripper control
    - Roll < -0.5: Activate gripper (attach nearby cubes)
    - Roll > 0.5: Deactivate gripper (release attached cubes)
    - Detects when cubes are placed in containers
    """
    
    def __init__(self, robot_id: int, end_effector_link_index: int = -1, wrist_roll_joint_index: int = 4):
        """
        Initialize the sticky cube system.
        
        Args:
            robot_id: PyBullet body ID of the robot
            end_effector_link_index: Link index of the end effector (-1 for base)
            wrist_roll_joint_index: Joint index of the wrist roll joint (used as gripper control)
        """
        self.robot_id = robot_id
        self.end_effector_link_index = end_effector_link_index
        self.wrist_roll_joint_index = wrist_roll_joint_index
        self.cubes: Dict[int, CubeState] = {}
        self.containers: Dict[int, Container] = {}
        self.attached_cube_id: Optional[int] = None
        
        # Physics parameters for attachment
        self.attachment_distance_threshold = 0.12  # 12cm (increased for easier interaction)
        self.constraint_force = 50.0  # Much lower force to allow joint movement
        self.constraint_damping = 0.5
        
        # Container detection parameters
        self.container_detection_threshold = 0.1  # 10cm
        
        # Gripper control parameters - back to wrist roll control with physics fixes
        self.gripper_attach_threshold = -0.3  # Roll angle to activate gripper (less negative)
        self.gripper_release_threshold = 0.2   # Roll angle to release gripper (lower positive)
        self.last_gripper_state = False  # Track gripper state changes
        self.last_roll_angle = 0.0  # Track roll angle changes
        
        # End effector attachment offset (relative to end effector link origin)
        # This positions the attachment point at the tip of the end effector
        self.ee_attachment_offset = [0.0, 0.0, 0.02]  # 2cm forward from end effector center
        
        print("🎮 Wrist Roll Gripper Controls:")
        print(f"   Roll < {self.gripper_attach_threshold:.1f}rad: Activate gripper (attach cubes)")
        print(f"   Roll > {self.gripper_release_threshold:.1f}rad: Release gripper (drop cubes)")
        
    def register_cube(self, cube_id: int) -> None:
        """Register a cube to be tracked by the system"""
        if cube_id not in self.cubes:
            pos, _ = p.getBasePositionAndOrientation(cube_id)
            self.cubes[cube_id] = CubeState(
                cube_id=cube_id,
                original_position=pos
            )
            
    def register_container(self, container_id: int, position: Tuple[float, float, float], 
                          size: Tuple[float, float, float], name: str) -> None:
        """Register a container that can hold cubes"""
        self.containers[container_id] = Container(
            container_id=container_id,
            position=position,
            size=size,
            name=name
        )
        
    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """Get the current position of the end effector tip (including offset)"""
        if self.end_effector_link_index == -1:
            # Use base position if no specific link specified
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            return pos
        else:
            # Get link world position and orientation
            link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            link_pos = link_state[0]  # World position
            link_orn = link_state[1]  # World orientation
            
            # Transform the attachment offset to world coordinates
            offset_world = p.multiplyTransforms(
                link_pos, link_orn,
                self.ee_attachment_offset, [0, 0, 0, 1]
            )[0]
            
            return offset_world
            
    def get_wrist_roll_angle(self) -> float:
        """Get the current wrist roll joint angle"""
        joint_state = p.getJointState(self.robot_id, self.wrist_roll_joint_index)
        return joint_state[0]  # Joint position in radians
        
    def is_gripper_active(self) -> bool:
        """Check if gripper should be active based on wrist roll angle"""
        roll_angle = self.get_wrist_roll_angle()
        return roll_angle < self.gripper_attach_threshold
        
    def should_release_gripper(self) -> bool:
        """Check if gripper should release based on wrist roll angle"""
        roll_angle = self.get_wrist_roll_angle()
        return roll_angle > self.gripper_release_threshold
            
    def get_distance_to_cube(self, cube_id: int) -> float:
        """Calculate distance between end effector and cube"""
        ee_pos = self.get_end_effector_position()
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        
        return math.sqrt(
            (ee_pos[0] - cube_pos[0])**2 +
            (ee_pos[1] - cube_pos[1])**2 +
            (ee_pos[2] - cube_pos[2])**2
        )
        
    def is_cube_in_container(self, cube_id: int) -> Optional[Container]:
        """Check if a cube is inside any container"""
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        
        for container in self.containers.values():
            # Simple box collision detection
            dx = abs(cube_pos[0] - container.position[0])
            dy = abs(cube_pos[1] - container.position[1])
            dz = abs(cube_pos[2] - container.position[2])
            
            if (dx <= container.size[0]/2 + self.container_detection_threshold and
                dy <= container.size[1]/2 + self.container_detection_threshold and
                dz <= container.size[2]/2 + self.container_detection_threshold):
                return container
                
        return None
        
    def attach_cube_to_end_effector(self, cube_id: int) -> bool:
        """
        Attach a cube to the end effector using a physics constraint.
        
        Args:
            cube_id: ID of the cube to attach
            
        Returns:
            True if attachment was successful, False otherwise
        """
        if self.attached_cube_id is not None:
            # Already have a cube attached
            return False
            
        if cube_id not in self.cubes:
            return False
            
        cube_state = self.cubes[cube_id]
        if cube_state.is_attached:
            return False
            
        try:
            # Create constraint - attach cube to specific point on end effector
            constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.end_effector_link_index,
                childBodyUniqueId=cube_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.ee_attachment_offset,  # Attach at tip of end effector
                childFramePosition=[0, 0, 0],   # Attach at cube center
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=[0, 0, 0, 1]
            )
            
            # Configure constraint properties with lower force to allow joint movement
            p.changeConstraint(constraint_id, maxForce=self.constraint_force)
            
            # Also reduce joint motor force when cube is attached to allow easier movement
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.wrist_roll_joint_index,
                controlMode=p.VELOCITY_CONTROL,
                force=0.1  # Very low force to allow manual movement
            )
            
            # Update cube state
            cube_state.is_attached = True
            cube_state.constraint_id = constraint_id
            self.attached_cube_id = cube_id
            
            return True
            
        except Exception as e:
            return False
            
    def detach_cube_from_end_effector(self, cube_id: int) -> bool:
        """
        Detach a cube from the end effector.
        
        Args:
            cube_id: ID of the cube to detach
            
        Returns:
            True if detachment was successful, False otherwise
        """
        if cube_id not in self.cubes:
            return False
            
        cube_state = self.cubes[cube_id]
        if not cube_state.is_attached or cube_state.constraint_id is None:
            return False
            
        try:
            # Remove the physics constraint
            p.removeConstraint(cube_state.constraint_id)
            
            # Update cube state
            cube_state.is_attached = False
            cube_state.constraint_id = None
            
            if self.attached_cube_id == cube_id:
                self.attached_cube_id = None
                
            # Restore normal joint motor control
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.wrist_roll_joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=10.0  # Normal force
            )
            
            return True
            
        except Exception as e:
            return False
            
    def update(self) -> None:
        """
        Update the sticky cube system. Call this every simulation step.
        
        This method:
        1. Checks wrist roll angle for gripper control
        2. Attaches cubes when gripper is active and cube is nearby
        3. Releases cubes when gripper is deactivated or cube is in container
        """
        # Get current state
        current_gripper_active = self.is_gripper_active()
        should_release = self.should_release_gripper()
        roll_angle = self.get_wrist_roll_angle()
        ee_pos = self.get_end_effector_position()
        
        # Update roll angle tracking
        self.last_roll_angle = roll_angle
        
        # No debug output in production version
        
        # Handle gripper activation (attach cube)
        if current_gripper_active and self.attached_cube_id is None:
            # Look for nearby cubes to attach
            closest_cube_id = None
            closest_distance = float('inf')
            
            for cube_id, cube_state in self.cubes.items():
                if not cube_state.is_attached:
                    distance = self.get_distance_to_cube(cube_id)
                    if distance <= self.attachment_distance_threshold and distance < closest_distance:
                        closest_cube_id = cube_id
                        closest_distance = distance
            
            # Attach the closest cube if found
            if closest_cube_id is not None:
                success = self.attach_cube_to_end_effector(closest_cube_id)
                if success:
                    print(f"🤏 Cube {closest_cube_id} attached to gripper")
        
        # Handle gripper deactivation (release cube)
        elif should_release and self.attached_cube_id is not None:
            print(f"✋ Cube {self.attached_cube_id} released from gripper")
            self.detach_cube_from_end_effector(self.attached_cube_id)
        
        # Also check for automatic release when cube is in container
        elif self.attached_cube_id is not None:
            container = self.is_cube_in_container(self.attached_cube_id)
            if container is not None:
                print(f"📦 Cube {self.attached_cube_id} placed in {container.name}")
                self.detach_cube_from_end_effector(self.attached_cube_id)
        
        # Update last gripper state
        self.last_gripper_state = current_gripper_active
                
    def get_status(self) -> Dict:
        """Get current status of the sticky cube system"""
        return {
            "attached_cube_id": self.attached_cube_id,
            "total_cubes": len(self.cubes),
            "attached_cubes": sum(1 for cube in self.cubes.values() if cube.is_attached),
            "containers": len(self.containers),
            "cube_states": {
                cube_id: {
                    "is_attached": state.is_attached,
                    "has_constraint": state.constraint_id is not None
                }
                for cube_id, state in self.cubes.items()
            }
        }
        
    def cleanup(self) -> None:
        """Clean up all constraints and reset the system"""
        for cube_state in self.cubes.values():
            if cube_state.is_attached and cube_state.constraint_id is not None:
                try:
                    p.removeConstraint(cube_state.constraint_id)
                except:
                    pass
                    
        self.cubes.clear()
        self.containers.clear()
        self.attached_cube_id = None
        print("🧹 Sticky cube system cleaned up")


def create_container_urdf(size: Tuple[float, float, float], position: Tuple[float, float, float], 
                         name: str = "container") -> int:
    """
    Create a container (open box) in the simulation.
    
    Args:
        size: (width, depth, height) of the container
        position: (x, y, z) position of the container
        name: Name for the container
        
    Returns:
        PyBullet body ID of the created container
    """
    # Create a simple box container using collision shapes
    # We'll create walls but leave the top open
    
    wall_thickness = 0.02
    width, depth, height = size
    x, y, z = position
    
    # Create container base
    base_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[width/2, depth/2, wall_thickness/2]
    )
    base_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[width/2, depth/2, wall_thickness/2],
        rgbaColor=[0.8, 0.6, 0.4, 1.0]  # Brown color
    )
    
    container_id = p.createMultiBody(
        baseMass=0,  # Static object
        baseCollisionShapeIndex=base_collision,
        baseVisualShapeIndex=base_visual,
        basePosition=[x, y, z - height/2 + wall_thickness/2]
    )
    
    # Create walls (4 sides)
    wall_positions = [
        [x + width/2 - wall_thickness/2, y, z - height/2 + height/2],  # Right wall
        [x - width/2 + wall_thickness/2, y, z - height/2 + height/2],  # Left wall  
        [x, y + depth/2 - wall_thickness/2, z - height/2 + height/2],  # Front wall
        [x, y - depth/2 + wall_thickness/2, z - height/2 + height/2],  # Back wall
    ]
    
    wall_sizes = [
        [wall_thickness/2, depth/2, height/2],  # Right wall
        [wall_thickness/2, depth/2, height/2],  # Left wall
        [width/2, wall_thickness/2, height/2],  # Front wall  
        [width/2, wall_thickness/2, height/2],  # Back wall
    ]
    
    for wall_pos, wall_size in zip(wall_positions, wall_sizes):
        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_size)
        wall_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=wall_size,
            rgbaColor=[0.6, 0.4, 0.2, 1.0]  # Darker brown for walls
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=wall_pos
        )
    
    return container_id