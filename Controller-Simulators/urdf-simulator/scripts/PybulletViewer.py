#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
import threading
from typing import Dict, List, Tuple

# Flask endpoints moved to scripts/http_api.py

try:
    import pybullet as p
    import pybullet_data
except ImportError as e:
    print("PyBullet is not installed in this environment. Activate your venv and run: pip install pybullet", file=sys.stderr)
    raise

# Local helpers
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from world_helpers import add_wall, spawn_cubes_on_floor, spawn_containers
from http_api import start_http_server
from apriltag_markers import (
    spawn_apriltags_on_table,
    spawn_apriltags_on_wall,
    list_apriltag_images,
    clear_tags_registry,
    write_tags_registry_json,
)


REVOLUTE = p.JOINT_REVOLUTE; PRISMATIC = p.JOINT_PRISMATIC
CONTROLLABLE_TYPES = {REVOLUTE, PRISMATIC}


def decode_name(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)


def connect(gui: bool) -> int:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")
    return cid


def setup_world(gravity: float, add_plane: bool) -> int:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)
    p.setTimeStep(1.0 / 240.0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 0.35],
    )
    plane_id = -1
    if add_plane:
        plane_id = p.loadURDF("plane.urdf")
        # Color the floor like a brown "table" surface
        try:
            p.changeVisualShape(plane_id, -1, rgbaColor=(0.45, 0.30, 0.20, 1.0))
        except Exception:
            pass
    return plane_id






def load_robot(urdf_path: str, fixed_base: bool, base_pos: List[float] = None, base_orn = None) -> int:
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if base_pos is None:
        base_pos = [0.0, 0.0, 0.0]
    if base_orn is None:
        base_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    print("############")
    print(os.path.join(pybullet_data.getDataPath()))
    print("############")
    #"kuka_iiwa/model_free_base.urdf"
    
    #pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
    #pandaUid = p.loadURDF("urdf/franka_panda/panda.urdf",
    pandaUid = p.loadURDF("urdf/five_dof_arm_with_gripper_.urdf",
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=bool(fixed_base),
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MERGE_FIXED_LINKS)
        
    #robot_id = p.loadURDF(
    #    urdf_path,
    #    basePosition=base_pos,
    #    baseOrientation=base_orn,
    #    useFixedBase=bool(fixed_base),
    #    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MERGE_FIXED_LINKS,
    #)
    return pandaUid


def get_controllable_joints(body_id: int) -> List[Tuple[int, str, int, float, float, float, float]]:
    """
    Returns list of tuples per joint:
    (jointIndex, jointName, jointType, lower, upper, maxForce, maxVelocity)
    Applies a hard cap for revolute joints to 180° total range ([-pi/2, +pi/2]).
    """
    joints: List[Tuple[int, str, int, float, float, float, float]] = []
    num = p.getNumJoints(body_id)
    cap_l, cap_u = -math.pi / 2.0, math.pi / 2.0  # 180° total
    for i in range(num):
        info = p.getJointInfo(body_id, i)
        j_type = info[2]
        if j_type in CONTROLLABLE_TYPES:
            j_index = info[0]
            j_name = decode_name(info[1])
            j_lower = float(info[8]) if info[8] is not None else 0.0
            j_upper = float(info[9]) if info[9] is not None else 0.0
            j_max_force = float(info[10]) if info[10] is not None else 50.0
            j_max_vel = float(info[11]) if info[11] is not None else 2.0

            if j_type == REVOLUTE:
                # If invalid bounds, default to cap; otherwise intersect with cap
                if not (j_upper > j_lower) or any(math.isinf(v) for v in (j_lower, j_upper)):
                    j_lower, j_upper = cap_l, cap_u
                else:
                    j_lower = max(cap_l, j_lower)
                    j_upper = min(cap_u, j_upper)
                    if not (j_upper > j_lower):
                        j_lower, j_upper = cap_l, cap_u

            joints.append((j_index, j_name, j_type, j_lower, j_upper, j_max_force, j_max_vel))
    return joints


def build_joints_index_map(joints: List[Tuple[int, str, int, float, float, float, float]]) -> Dict[int, Dict[str, float]]:
    """
    Build a metadata map per joint index for HTTP exposure.
    """
    mapping: Dict[int, Dict[str, float]] = {}
    for (j_index, j_name, j_type, lower, upper, j_max_force, j_max_vel) in joints:
        mapping[j_index] = {
            "name": j_name,
            "type": int(j_type),
            "lower": float(lower),
            "upper": float(upper),
            "maxForce": float(j_max_force),
            "maxVelocity": float(j_max_vel),
        }
    return mapping


# HTTP server moved to scripts/http_api.py


def create_joint_sliders(joints: List[Tuple[int, str, int, float, float, float, float]]) -> Dict[int, int]:
    """
    Create debug sliders for joints. Returns mapping: jointIndex -> sliderParamId
    For revolute joints, slider range is capped to 180° total ([-pi/2, +pi/2]).
    For prismatic with invalid/identical bounds, default to [-0.2, 0.2].
    """
    mapping: Dict[int, int] = {}
    cap_l, cap_u = -math.pi / 2.0, math.pi / 2.0
    for (j_index, j_name, j_type, j_lower, j_upper, _j_force, _j_vel) in joints:
        if j_type == REVOLUTE:
            lower = max(cap_l, j_lower)
            upper = min(cap_u, j_upper)
            if not (upper > lower) or any(math.isinf(v) for v in (lower, upper)):
                lower, upper = cap_l, cap_u
            default = 0.0
            pid = p.addUserDebugParameter(f"{j_name} (idx {j_index})", lower, upper, default)
        else:
            # PRISMATIC
            lower, upper = j_lower, j_upper
            if not (upper > lower) or any(math.isinf(v) for v in (lower, upper)):
                lower, upper = -0.2, 0.2
            default = 0.0
            pid = p.addUserDebugParameter(f"{j_name} (idx {j_index}) [m]", lower, upper, default)
        mapping[j_index] = pid
    return mapping


def drive_joints_position(
    body_id: int,
    joints: List[Tuple[int, str, int, float, float, float, float]],
    slider_map: Dict[int, int],
    http_targets: Dict[int, float],
    lock: threading.Lock,
) -> None:
    for (j_index, _j_name, _j_type, _l, _u, j_max_force, j_max_vel) in joints:
        target = 0.0
        override = None
        if lock is not None:
            with lock:
                override = http_targets.get(j_index)
        if override is not None:
            target = float(override)
        elif slider_map and (j_index in slider_map):
            target = p.readUserDebugParameter(slider_map[j_index])

        p.setJointMotorControl2(
            bodyUniqueId=body_id,
            jointIndex=j_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=max(1.0, j_max_force if j_max_force and not math.isinf(j_max_force) else 50.0),
            maxVelocity=max(0.1, j_max_vel if j_max_vel and not math.isinf(j_max_vel) else 2.0),
        )


def run_loop(
    body_id: int,
    joints: List[Tuple[int, str, int, float, float, float, float]],
    realtime: bool,
    use_sliders: bool,
    http_targets: Dict[int, float],
    lock: threading.Lock,
) -> None:
    slider_map = create_joint_sliders(joints) if use_sliders else {}
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if use_sliders else 0)
    p.setRealTimeSimulation(1 if realtime else 0)

    try:
        while True:
            drive_joints_position(body_id, joints, slider_map, http_targets, lock)
            
            if not realtime:
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
            else:
                time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyBullet URDF Viewer with Joint Sliders and HTTP control")
    parser.add_argument("--urdf", type=str, default=os.path.join("urdf", "five_dof_arm.urdf"), help="Path to URDF file")
    parser.add_argument("--gui", action="store_true", default=True, help="Use GUI connection (default)")
    parser.add_argument("--direct", dest="gui", action="store_false", help="Use DIRECT (headless) connection")
    parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity along Z axis")
    parser.add_argument("--no-plane", action="store_true", help="Do not add plane.urdf ground")
    parser.add_argument("--fixed-base", action="store_true", default=True, help="Load robot with fixed base (default)")
    parser.add_argument("--realtime", action="store_true", default=True, help="Use real-time simulation (default)")
    parser.add_argument("--wall-mount", action="store_true", default=True, help="Mount robot on vertical wall at x=--wall-x and add wall plane (default)")
    parser.add_argument("--no-wall-mount", dest="wall_mount", action="store_false", help="Disable wall mount (mount on ground instead)")
    parser.add_argument("--wall-x", type=float, default=0.0, help="X position of wall plane when --wall-mount")
    parser.add_argument("--base-height", type=float, default=1.35, help="Base height (z) when wall-mounted")
    parser.add_argument("--cube-count", type=int, default=3, help="Number of cubes to drop on the floor (0 to disable)")
    parser.add_argument("--container-count", type=int, default=2, help="Number of containers to spawn (0 to disable)")

    # HTTP API options
    parser.add_argument("--http-host", type=str, default="127.0.0.1", help="HTTP API bind host (default: 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=5000, help="HTTP API bind port (default: 5000)")
    parser.add_argument("--no-http", action="store_true", help="Disable HTTP API server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cid = connect(gui=args.gui)
    try:
        setup_world(gravity=args.gravity, add_plane=not args.no_plane)

        # Wall-mounted orientation and placement
        if args.wall_mount:
            add_wall(args.wall_x)
            #os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
            # Add table below the robot (table surface at 0.6m height)
            # Load table rotated 90 degrees around Z so the long edge aligns correctly
            table_yaw = math.pi / 2.0
            table_orn = p.getQuaternionFromEuler([0, 0, table_yaw])
            table_pos = [float(args.wall_x), 0.0, 0.0]
            table_id = p.loadURDF("table/table.urdf", basePosition=table_pos, baseOrientation=table_orn)

            # Snap table fully in front of the wall plane (x = wall_x) with a small clearance
            try:
                clearance = 0.01
                aabb_min, aabb_max = p.getAABB(table_id)
                x_min = aabb_min[0]
                desired_x_min = float(args.wall_x) + clearance
                dx = desired_x_min - x_min
                if abs(dx) > 1e-6:
                    base_pos_now, base_orn_now = p.getBasePositionAndOrientation(table_id)
                    new_pos = [base_pos_now[0] + dx, base_pos_now[1], base_pos_now[2]]
                    p.resetBasePositionAndOrientation(table_id, new_pos, base_orn_now)
            except Exception as e:
                print(f"Failed to reposition table against wall: {e}", file=sys.stderr)

            # Place AprilTags after final table pose
            try:
                # Reset registry for this run
                clear_tags_registry()

                # Ensure unique tags on the table. Use as many as available (up to 9 positions).
                images = list_apriltag_images()
                table_positions_needed = 9
                table_img_count = min(len(images), table_positions_needed)
                table_images = images[:table_img_count]
                spawn_apriltags_on_table(table_id, image_files=table_images)

                # Optionally place tags on the wall too, using remaining unique images if any (up to 6 positions).
                wall_positions_needed = 6
                remaining = images[table_img_count:]
                wall_img_count = min(len(remaining), wall_positions_needed)
                if wall_img_count > 0:
                    wall_images = remaining[:wall_img_count]
                    spawn_apriltags_on_wall(args.wall_x, table_id, image_files=wall_images)
                else:
                    print("No remaining unique AprilTag images for wall placement; skipped wall tags.")

                # Export world-space AprilTag mapping JSON for downstream calibration (Option B)
                tags_json_out = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "3d-april-tags-coordinate-test", "tags_world.json"))
                try:
                    os.makedirs(os.path.dirname(tags_json_out), exist_ok=True)
                    write_tags_registry_json(tags_json_out)
                    print(f"Exported AprilTags world mapping to: {tags_json_out}")
                except Exception as ex:
                    print(f"Failed to write AprilTags world mapping JSON: {ex}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to spawn AprilTags: {e}", file=sys.stderr)

            base_pos = [float(args.wall_x), 0.0, float(args.base_height)]
            base_orn = p.getQuaternionFromEuler([0, math.pi / 2.0, 0])  # Z-up -> X-up (onto wall)
        else:
            base_pos = [0.0, 0.0, 0.0]
            base_orn = p.getQuaternionFromEuler([0, 0, 0])

        robot_id = load_robot(args.urdf, fixed_base=args.fixed_base, base_pos=base_pos, base_orn=base_orn)

        joints = get_controllable_joints(robot_id)
        if not joints:
            print("No controllable joints found in the URDF. Exiting.", file=sys.stderr)
            return

        print(f"Controllable joints: {len(joints)}")

        # Prepare HTTP control
        http_targets: Dict[int, float] = {}
        targets_lock = threading.Lock()
        joints_info_map = build_joints_index_map(joints)

        if not args.no_http:
            start_http_server(args.http_host, args.http_port, http_targets, joints_info_map, targets_lock, robot_id)
            print(f"HTTP API at http://{args.http_host}:{args.http_port} (GET /api/health, /api/joints, /api/joint/<id>/<value>[/instant], /api/camera/<id>)")

        # Drop cubes on the floor (if floor is present)
        cube_ids = []
        if (not args.no_plane) and args.cube_count and args.cube_count > 0:
            cube_ids = spawn_cubes_on_floor(args.cube_count)

        # Spawn containers
        container_info = []
        if args.container_count and args.container_count > 0:
            container_info = spawn_containers(args.container_count)

        run_loop(
            robot_id,
            joints,
            realtime=args.realtime,
            use_sliders=args.gui,
            http_targets=http_targets,
            lock=targets_lock,
        )
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()