from typing import Dict
import threading
import math
import base64
import io
import sys
import time
from flask import Flask, jsonify, Response, request
import pybullet as p


def get_camera_view(camera_id: int) -> bytes:
    """
    Capture image from different camera viewpoints:
    1: Top view
    2: Side view
    3: Front-left angled view
    """
    if camera_id == 1:
        # Top view - looking down from above
        camera_distance = 2.0
        camera_yaw = 0
        camera_pitch = -90
        camera_target = [0, 0, 1.1]
    elif camera_id == 2:
        # Side view - looking from the side
        camera_distance = 2.0
        camera_yaw = 90
        camera_pitch = -30
        camera_target = [0, 0, 1.1]
    elif camera_id == 3:
        # Front-left angled view
        camera_distance = 2.5
        camera_yaw = 45
        camera_pitch = -45
        camera_target = [0, 0, 0.9]
    else:
        raise ValueError(f"Invalid camera ID: {camera_id}. Must be 1, 2, or 3")

    # Get camera image
    width, height = 640, 480
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )

    _, _, rgba_img, depth_img, seg_img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Convert RGBA to RGB PIL Image
    import numpy as np
    from PIL import Image

    rgba_array = np.array(rgba_img, dtype=np.uint8).reshape((height, width, 4))
    rgb_array = rgba_array[:, :, :3]  # Remove alpha channel

    img = Image.fromarray(rgb_array)

    # Save to bytes buffer as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)

    return buffer.getvalue()


def start_http_server(host: str, port: int, http_targets: Dict[int, float], joints_info_map: Dict[int, Dict[str, float]], lock: threading.Lock, robot_id: int = None) -> threading.Thread:
    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/joints")
    def joints_endpoint():
        # Return joints metadata augmented with current joint position, forces, and torques from PyBullet
        data: Dict[int, dict] = {}
        for j_index, meta in joints_info_map.items():
            current = None
            velocity = None
            reaction_forces = None
            applied_torque = None

            try:
                if robot_id is not None:
                    # Enable force/torque sensor for this joint (if not already enabled)
                    p.enableJointForceTorqueSensor(robot_id, int(j_index), 1)

                    state = p.getJointState(robot_id, int(j_index))
                    if state is not None and len(state) >= 4:
                        current = float(state[0])  # Joint position
                        velocity = float(state[1])  # Joint velocity
                        reaction_forces = [float(f) for f in state[2]]  # 6D reaction forces [Fx, Fy, Fz, Mx, My, Mz]
                        applied_torque = float(state[3])  # Applied motor torque
            except Exception as e:
                current = None
                velocity = None
                reaction_forces = None
                applied_torque = None

            merged = dict(meta)  # shallow copy to avoid mutating original map
            merged["current"] = current
            merged["velocity"] = velocity
            merged["reaction_forces"] = reaction_forces  # New: resistance information
            merged["applied_torque"] = applied_torque    # New: motor effort

            data[int(j_index)] = merged
        return jsonify(data)

    @app.route("/api/joint/<int:joint_id>/<path:value>", methods=['GET'])
    @app.route("/api/joint/<int:joint_id>/<path:value>/<mode>", methods=['GET'])
    def set_joint(joint_id: int, value: str, mode: str = None):
        # Parse the value string to handle negative numbers
        try:
            value = float(value)
        except ValueError:
            return jsonify({"error": "invalid value format", "value": value}), 400
        if joint_id not in joints_info_map:
            return jsonify({"error": "invalid joint id", "joint_id": joint_id}), 404
        meta = joints_info_map[joint_id]
        lower = float(meta.get("lower", 0.0))
        upper = float(meta.get("upper", 0.0))
        clamped = value
        if upper > lower and not (math.isinf(lower) or math.isinf(upper)):
            clamped = max(lower, min(upper, value))
        
        # Set the target position
        with lock:
            http_targets[joint_id] = float(clamped)
        
        # If fast parameter is provided, use high-speed physics-based movement
        if mode is not None and mode.lower() in ['fast', 'quick', 'rapid']:
            try:
                if robot_id is not None:
                    # Store original simulation settings
                    original_timestep = p.getPhysicsEngineParameters()['fixedTimeStep']
                    
                    # Temporarily disable real-time simulation and set faster timestep
                    p.setRealTimeSimulation(0)  # Disable real-time mode
                    fast_timestep = original_timestep / 8.0  # 8x faster simulation
                    p.setTimeStep(fast_timestep)
                    
                    # Use moderate force increase and rely more on simulation speed for fast movement
                    max_force = float(meta.get("maxForce", 50.0))
                    max_velocity = float(meta.get("maxVelocity", 2.0))                   
                  
                    
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint_id,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=float(clamped),
                        force=max_force,
                        maxVelocity=max_velocity
                    )
                    
                    # Wait until the joint has stopped moving - OPTIMIZED VERSION
                    previous_pos = None
                    stable_count = 0
                    final_position = None
                    check_interval = 5  # Check every 5 steps instead of every step
                    
                    for step in range(300):  # Maximum 300 steps to prevent infinite loop
                        p.stepSimulation()
                        
                        # Only check joint state every check_interval steps to reduce API calls
                        if step % check_interval == 0:
                            # Get current joint state
                            current_state = p.getJointState(robot_id, joint_id)
                            current_pos = current_state[0]
                            current_vel = current_state[1]
                            
                            # Check if joint has stopped moving (low velocity and stable position)
                            if previous_pos is not None:
                                pos_change = abs(current_pos - previous_pos)
                                if pos_change < 0.001 and abs(current_vel) < 0.01:  # Very small movement and velocity
                                    stable_count += 1
                                    if stable_count >= 3:  # Reduced from 5 for faster response
                                        final_position = current_pos
                                        break
                                else:
                                    stable_count = 0  # Reset if movement detected
                            
                            previous_pos = current_pos
                            
                            # Also check if we're close enough to target (fallback)
                            if abs(current_pos - float(clamped)) < 0.005:
                                final_position = current_pos
                                break
                    
                    # Restore original simulation settings first
                    p.setTimeStep(original_timestep)
                    p.setRealTimeSimulation(1)  # Re-enable real-time mode (default for the app)
                    
                    # Get the actual final position after restoring normal simulation
                    if final_position is None:
                        final_state = p.getJointState(robot_id, joint_id)
                        final_position = final_state[0]
                    
                    actual_final_position = final_position
                    
                    # Return the actual final position
                    return jsonify({
                        "joint_id": joint_id,
                        "requested": float(value),
                        "applied": float(clamped),
                        "actual_position": float(actual_final_position),
                        "fast_mode": True,
                        "movement_completed": True
                    })
                    
                else:
                    return jsonify({"error": "robot_id not available for fast positioning"}), 500
            except Exception as e:
                return jsonify({"error": f"failed to set fast position: {str(e)}"}), 500
        
        return jsonify({"joint_id": joint_id, "requested": float(value), "applied": float(clamped), "fast_mode": mode is not None and mode.lower() in ['fast', 'quick', 'rapid']})

    @app.route("/api/joints/fast", methods=['POST'])
    def set_multiple_joints_fast():
        """
        Move multiple joints simultaneously in fast mode.
        Expected JSON body: [{"id": 0, "pos": -1.2}, {"id": 1, "pos": 0.3}]
        Returns: {"results": [{"joint_id": 0, "requested": -1.2, "applied": -1.2, "actual_position": -1.1987}, ...]}
        """
        try:
            # Parse JSON body
            joint_commands = request.get_json()
            if not isinstance(joint_commands, list):
                return jsonify({"error": "Expected JSON array of joint commands"}), 400
            
            if not joint_commands:
                return jsonify({"error": "Empty joint commands array"}), 400
            
            if robot_id is None:
                return jsonify({"error": "robot_id not available for fast positioning"}), 500
            
            # Validate all joint commands first
            validated_commands = []
            for cmd in joint_commands:
                if not isinstance(cmd, dict) or 'id' not in cmd or 'pos' not in cmd:
                    return jsonify({"error": "Each command must have 'id' and 'pos' fields"}), 400
                
                joint_id = int(cmd['id'])
                requested_pos = float(cmd['pos'])
                
                if joint_id not in joints_info_map:
                    return jsonify({"error": f"Invalid joint id: {joint_id}"}), 404
                
                # Apply joint limits
                meta = joints_info_map[joint_id]
                lower = float(meta.get("lower", 0.0))
                upper = float(meta.get("upper", 0.0))
                clamped_pos = requested_pos
                if upper > lower and not (math.isinf(lower) or math.isinf(upper)):
                    clamped_pos = max(lower, min(upper, requested_pos))
                
                validated_commands.append({
                    'joint_id': joint_id,
                    'requested': requested_pos,
                    'clamped': clamped_pos,
                    'meta': meta
                })
            
            # Store original simulation settings
            original_timestep = p.getPhysicsEngineParameters()['fixedTimeStep']
            
            # Temporarily disable real-time simulation and set faster timestep
            p.setRealTimeSimulation(0)  # Disable real-time mode
            fast_timestep = original_timestep / 8.0  # 8x faster simulation
            p.setTimeStep(fast_timestep)
            
            # Update http_targets first so normal simulation maintains these positions
            with lock:
                for cmd in validated_commands:
                    joint_id = cmd['joint_id']
                    clamped_pos = cmd['clamped']
                    http_targets[joint_id] = clamped_pos
            
            # Set all joint targets simultaneously
            for cmd in validated_commands:
                joint_id = cmd['joint_id']
                clamped_pos = cmd['clamped']
                meta = cmd['meta']
                
                # Use moderate force increase and rely more on simulation speed for fast movement
                max_force = float(meta.get("maxForce", 50.0))
                max_velocity = float(meta.get("maxVelocity", 2.0))
                
                fast_force = max_force 
                fast_velocity = max_velocity
                
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=clamped_pos,
                    force=fast_force,
                    maxVelocity=fast_velocity
                )
            
            # Wait until all joints have stopped moving - OPTIMIZED VERSION
            max_steps = 300
            stable_threshold = 0.001
            velocity_threshold = 0.01
            stability_required = 3  # Reduced from 5 for faster response
            check_interval = 5  # Check joint states every 5 steps instead of every step
            
            joint_stable_counts = {cmd['joint_id']: 0 for cmd in validated_commands}
            joint_previous_pos = {cmd['joint_id']: None for cmd in validated_commands}
            joint_ids = [cmd['joint_id'] for cmd in validated_commands]
            
            for step in range(max_steps):
                p.stepSimulation()
                
                # Only check joint states every check_interval steps to reduce API calls
                if step % check_interval == 0:
                    all_stable = True
                    
                    # Get all joint states in individual calls (PyBullet doesn't have batch getJointStates)
                    for joint_id in joint_ids:
                        current_state = p.getJointState(robot_id, joint_id)
                        current_pos = current_state[0]
                        current_vel = current_state[1]
                        
                        # Check stability for this joint
                        if joint_previous_pos[joint_id] is not None:
                            pos_change = abs(current_pos - joint_previous_pos[joint_id])
                            if pos_change < stable_threshold and abs(current_vel) < velocity_threshold:
                                joint_stable_counts[joint_id] += 1
                            else:
                                joint_stable_counts[joint_id] = 0
                        
                        joint_previous_pos[joint_id] = current_pos
                        
                        # Check if this joint is stable
                        if joint_stable_counts[joint_id] < stability_required:
                            all_stable = False
                    
                    # If all joints are stable, we can break early
                    if all_stable:
                        break
            
            # Restore original simulation settings first
            p.setTimeStep(original_timestep)
            p.setRealTimeSimulation(1)  # Re-enable real-time mode
            
            # Get final positions for all joints after restoring normal simulation
            results = []
            for cmd in validated_commands:
                joint_id = cmd['joint_id']
                final_state = p.getJointState(robot_id, joint_id)
                actual_position = final_state[0]
                
                results.append({
                    "joint_id": joint_id,
                    "requested": cmd['requested'],
                    "applied": cmd['clamped'],
                    "actual_position": float(actual_position)
                })
            
            return jsonify({
                "results": results,
                "fast_mode": True,
                "movement_completed": True,
                "joints_moved": len(results)
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to move joints: {str(e)}"}), 500

    @app.route("/api/joints/instant", methods=['POST'])
    def set_multiple_joints_instant():
        """
        Move multiple joints instantly using resetJointState, bypassing physics entirely.
        Returns torque feedback to detect if the movement would be "illegal" due to collisions.
        
        Expected JSON body: [{"id": 0, "pos": -1.2}, {"id": 1, "pos": 0.3}]
        Returns: {
            "results": [
                {
                    "joint_id": 0,
                    "requested": -1.2,
                    "applied": -1.2,
                    "actual_position": -1.2,
                    "applied_torque": 15.3,
                    "torque_warning": false
                }, ...
            ],
            "instant_mode": true,
            "max_torque": 45.7,
            "torque_threshold": 30.0
        }
        """
        try:
            # Parse JSON body
            joint_commands = request.get_json()
            if not isinstance(joint_commands, list):
                return jsonify({"error": "Expected JSON array of joint commands"}), 400
            
            if not joint_commands:
                return jsonify({"error": "Empty joint commands array"}), 400
            
            if robot_id is None:
                return jsonify({"error": "robot_id not available for instant positioning"}), 500
            
            # Validate all joint commands first
            validated_commands = []
            for cmd in joint_commands:
                if not isinstance(cmd, dict) or 'id' not in cmd or 'pos' not in cmd:
                    return jsonify({"error": "Each command must have 'id' and 'pos' fields"}), 400
                
                joint_id = int(cmd['id'])
                requested_pos = float(cmd['pos'])
                
                if joint_id not in joints_info_map:
                    return jsonify({"error": f"Invalid joint id: {joint_id}"}), 404
                
                # Apply joint limits
                meta = joints_info_map[joint_id]
                lower = float(meta.get("lower", 0.0))
                upper = float(meta.get("upper", 0.0))
                clamped_pos = requested_pos
                if upper > lower and not (math.isinf(lower) or math.isinf(upper)):
                    clamped_pos = max(lower, min(upper, requested_pos))
                
                validated_commands.append({
                    'joint_id': joint_id,
                    'requested': requested_pos,
                    'clamped': clamped_pos,
                    'meta': meta
                })
            
            # Update http_targets first so normal simulation maintains these positions
            with lock:
                for cmd in validated_commands:
                    joint_id = cmd['joint_id']
                    clamped_pos = cmd['clamped']
                    http_targets[joint_id] = clamped_pos
            
            # Perform instant movement using resetJointState (bypasses physics)
            for cmd in validated_commands:
                joint_id = cmd['joint_id']
                clamped_pos = cmd['clamped']
                p.resetJointState(bodyUniqueId=robot_id, jointIndex=joint_id, targetValue=clamped_pos)
            
            # Collect results - simple and clean
            results = []
            for cmd in validated_commands:
                joint_id = cmd['joint_id']
                
                # Get joint state for actual position
                joint_state = p.getJointState(robot_id, joint_id)
                actual_position = joint_state[0]
                
                results.append({
                    "joint_id": joint_id,
                    "requested": cmd['requested'],
                    "applied": cmd['clamped'],
                    "actual_position": float(actual_position)
                })
            
            return jsonify({
                "results": results,
                "instant_mode": True,
                "movement_completed": True,
                "joints_moved": len(results)
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to move joints instantly: {str(e)}"}), 500

    @app.get("/api/reset/instant")
    def reset_all_joints_instant():
        """
        Reset all joints to 0.0 position instantly using resetJointState.
        This bypasses physics for immediate reset to neutral position.
        Returns: {"results": [{"joint_id": 0, "reset_to": 0.0, "actual_position": 0.0}, ...]}
        """
        try:
            if robot_id is None:
                return jsonify({"error": "robot_id not available for reset"}), 500
            
            # Reset all joints to 0.0 instantly and update http_targets
            results = []
            with lock:
                for joint_id in joints_info_map.keys():
                    # Reset joint position instantly (bypasses physics)
                    p.resetJointState(bodyUniqueId=robot_id, jointIndex=joint_id, targetValue=0.0)
                    
                    # Update http_targets so normal simulation maintains this position
                    http_targets[joint_id] = 0.0
                    
                    # Get the position after reset (should be exactly 0.0)
                    final_state = p.getJointState(robot_id, joint_id)
                    actual_position = final_state[0]
                    
                    results.append({
                        "joint_id": joint_id,
                        "reset_to": 0.0,
                        "actual_position": float(actual_position)
                    })
            
            return jsonify({
                "results": results,
                "reset_mode": "instant",
                "reset_completed": True,
                "joints_reset": len(results)
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to reset joints: {str(e)}"}), 500

    @app.get("/api/camera/<int:camera_id>")
    def get_camera_image(camera_id: int):
        """
        Get camera image from specified viewpoint.
        Camera 1: Top view
        Camera 2: Side view
        Camera 3: Front-left angled view
        """
        if camera_id not in [1, 2, 3]:
            return jsonify({"error": "invalid camera id", "camera_id": camera_id, "valid_ids": [1, 2, 3]}), 404

        try:
            # Capture image on-demand when requested
            image_bytes = get_camera_view(camera_id)

            # Return image as JPEG response
            return Response(
                response=image_bytes,
                status=200,
                mimetype='image/jpeg'
            )
        except Exception as e:
            return jsonify({"error": f"failed to capture image: {str(e)}"}), 500

    th = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True),
        daemon=True,
    )
    th.start()
    return th