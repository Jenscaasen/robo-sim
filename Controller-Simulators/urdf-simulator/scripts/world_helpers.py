from typing import List, Tuple
import math
import random


def add_wall(x: float = 0.0, rgba=(0.6, 0.6, 0.6, 1.0)) -> int:
    import pybullet as p
    quat = p.getQuaternionFromEuler([0, math.pi / 2.0, 0])
    wid = p.loadURDF("plane.urdf", basePosition=[x, 0.0, 0.0], baseOrientation=quat)
    try:
        p.changeVisualShape(wid, -1, rgbaColor=rgba)
    except Exception:
        pass
    return wid


def spawn_cubes_on_floor(count: int, x: float = 0.5, y0: float = 0.0, z: float = 0.65, spacing: float = 0.15) -> List[int]:
    import pybullet as p
    ids: List[int] = []
    palette = [
        (0.90, 0.10, 0.10, 1.0),  # red
        (0.10, 0.60, 0.95, 1.0),  # blue
        (0.15, 0.80, 0.35, 1.0),  # green
        (0.95, 0.85, 0.10, 1.0),  # yellow
        (0.80, 0.20, 0.85, 1.0),  # magenta
        (1.00, 0.55, 0.00, 1.0),  # orange
    ]
    for i in range(max(0, count)):
        y = y0 + (i - (count - 1) * 0.5) * spacing
        uid = p.loadURDF("cube_small.urdf", basePosition=[x, y, z])
        color = palette[i % len(palette)]
        try:
            p.changeVisualShape(uid, -1, rgbaColor=color)
        except Exception:
            pass
        ids.append(uid)
    return ids


def create_container(size: Tuple[float, float, float], position: Tuple[float, float, float],
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
    import pybullet as p
    
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


def spawn_containers(count: int = 2, base_x: float = 1.2, base_y: float = 0.0,
                    spacing: float = 0.4) -> List[Tuple[int, str]]:
    """
    Spawn containers in the simulation.
    
    Args:
        count: Number of containers to spawn
        base_x: Base X position for first container
        base_y: Base Y position for containers
        spacing: Spacing between containers
        
    Returns:
        List of tuples (container_id, container_name)
    """
    containers = []
    container_size = (0.2, 0.2, 0.15)  # 20cm x 20cm x 15cm
    
    for i in range(count):
        y_pos = base_y + (i - (count - 1) * 0.5) * spacing
        position = (base_x, y_pos, container_size[2]/2)  # Place on ground
        name = f"Container_{i+1}"
        
        container_id = create_container(container_size, position, name)
        containers.append((container_id, name))
        
    return containers