from typing import Dict
import threading
import math
import base64
import io
import sys
from flask import Flask, jsonify, Response
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
        camera_target = [0, 0, 0.5]
    elif camera_id == 2:
        # Side view - looking from the side
        camera_distance = 2.0
        camera_yaw = 90
        camera_pitch = -30
        camera_target = [0, 0, 0.5]
    elif camera_id == 3:
        # Front-left angled view
        camera_distance = 2.5
        camera_yaw = 45
        camera_pitch = -45
        camera_target = [0, 0, 0.3]
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
        # Return joints metadata augmented with current joint position from PyBullet
        data: Dict[int, dict] = {}
        for j_index, meta in joints_info_map.items():
            current = None
            try:
                if robot_id is not None:
                    state = p.getJointState(robot_id, int(j_index))
                    if state is not None:
                        current = float(state[0])
            except Exception:
                current = None
            merged = dict(meta)  # shallow copy to avoid mutating original map
            merged["current"] = current
            data[int(j_index)] = merged
        return jsonify(data)

    @app.route("/api/joint/<int:joint_id>/<path:value>", methods=['GET'])
    @app.route("/api/joint/<int:joint_id>/<path:value>/<instant>", methods=['GET'])
    def set_joint(joint_id: int, value: str, instant: str = None):
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
        
        # If instant parameter is provided, immediately set the joint position
        if instant is not None and instant.lower() in ['true', '1', 'instant']:
            try:
                if robot_id is not None:
                    p.resetJointState(bodyUniqueId=robot_id, jointIndex=joint_id, targetValue=float(clamped))
                else:
                    return jsonify({"error": "robot_id not available for instant positioning"}), 500
            except Exception as e:
                return jsonify({"error": f"failed to set instant position: {str(e)}"}), 500
        
        return jsonify({"joint_id": joint_id, "requested": float(value), "applied": float(clamped), "instant": instant is not None})

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