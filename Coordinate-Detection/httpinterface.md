# Robot Arm HTTP Interface

This document describes the HTTP endpoints used to control the simulator robot arm, inspect joint states, reset the world, and retrieve camera images. It is intended as a reference for building a reinforcement learning environment that positions the robot arm above a red cube.

Base URL (local):
- http://127.0.0.1:5000

Content types:
- Requests: application/json (for POST bodies)
- Images: image/jpeg (for camera endpoints)

Authentication:
- None (local development)



## 1) Joints API

### 1.1 Get Joint States

GET /api/joints

Returns a JSON object mapping joint indices (as string keys) to their state, including limits, current position, and dynamics information.

Example response (truncated):
{
  "0": {
    "applied_torque": 0.0,
    "current": 0.0,
    "lower": -1.5707963267948966,
    "maxForce": 50.0,
    "maxVelocity": 2.0,
    "name": "shoulder_yaw",
    "reaction_forces": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "type": 0,
    "upper": 1.5707963267948966,
    "velocity": 0.0
  },
  "1": { ... "name": "shoulder_pitch", ... },
  "2": { ... "name": "elbow_pitch", ... },
  "3": { ... "name": "wrist_roll_1", ... },
  "4": { ... "name": "wrist_yaw", ... },
  "5": { ... "name": "panda_finger_joint1", "type": 1, "lower": 0.0, "upper": 0.04, ... },
  "6": { ... "name": "panda_finger_joint2", "type": 1, "lower": 0.0, "upper": 0.04, ... }
}

Notes:
- Joint indices are 0-based.
- Units:
  - Positions are in radians for rotational joints (type 0).
  - Positions are in meters for finger prismatic joints (type 1).
  - Velocity in rad/s or m/s accordingly.
  - Torques/forces in engine units (sim-specific).
- Limits:
  - lower and upper specify the valid range for pos commands.
- Typical control set for positioning task:
  - Use joints 0..4 for arm positioning (ignore 5 & 6 unless you control gripper width).

cURL:
curl -s http://127.0.0.1:5000/api/joints | jq .

Python (requests):
import requests

resp = requests.get("http://127.0.0.1:5000/api/joints", timeout=2.0)
resp.raise_for_status()
joints = resp.json()
print({k: (v.get("name"), v.get("current")) for k, v in joints.items()})


### 1.2 Set Joint Positions

POST /api/joints
Content-Type: application/json
Accept: application/json

Body: an array of objects { "id": int, "pos": float }, where pos is the absolute target position in the joint’s units and must be within [lower, upper].

Example body:
[
  {"id": 0, "pos": 0.10},
  {"id": 1, "pos": 0.90},
  {"id": 2, "pos": 0.60},
  {"id": 3, "pos": 0.20},
  {"id": 4, "pos": 0.00}
]

Response:
- 200 OK on success (typically with an acknowledgment JSON or empty body)
- 400 Bad Request if payload is invalid or positions out of bounds

Guidelines:
- For RL, prefer incremental updates (delta moves) on top of the last known position to stay within velocity constraints and avoid large jumps.
- Always clamp to [lower, upper] using values from GET /api/joints.

cURL:
curl -X POST http://127.0.0.1:5000/api/joints \
  -H "Content-Type: application/json" \
  -d '[{"id":0,"pos":0.1},{"id":1,"pos":0.9},{"id":2,"pos":0.6},{"id":3,"pos":0.2},{"id":4,"pos":0.0}]'

Python (requests):
import requests

cmd = [
  {"id": 0, "pos": 0.10},
  {"id": 1, "pos": 0.90},
  {"id": 2, "pos": 0.60},
  {"id": 3, "pos": 0.20},
  {"id": 4, "pos": 0.00},
]
resp = requests.post("http://127.0.0.1:5000/api/joints", json=cmd, timeout=2.0)
resp.raise_for_status()


### 1.3 Reset Joints (Instant)

GET /api/reset/instant

Resets the robot to a default initial state instantly.

Response:
- 200 OK on success

cURL:
curl -X GET http://127.0.0.1:5000/api/reset/instant

Python (requests):
import requests

requests.get("http://127.0.0.1:5000/api/reset/instant", timeout=2.0).raise_for_status()



## 2) Camera API

Three cameras are available. All return image/jpeg. Provide Accept: image/jpeg for clarity.

Common:
- GET /api/camera/{id}
- id: 1, 2, or 3

### 2.1 Camera 1
GET /api/camera/1
Description: Returns a camera image (general purpose view).

cURL:
curl -s -H "Accept: image/jpeg" -o cam1.jpg http://127.0.0.1:5000/api/camera/1

Python (OpenCV):
import cv2, numpy as np, requests

img_bytes = requests.get("http://127.0.0.1:5000/api/camera/1", headers={"Accept":"image/jpeg"}, timeout=2.0).content
image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
print(image.shape)


### 2.2 Camera 2 (side view)
GET /api/camera/2
Description: Side view camera image, useful for height/vertical alignment checks.

cURL:
curl -s -H "Accept: image/jpeg" -o cam2.jpg http://127.0.0.1:5000/api/camera/2


### 2.3 Camera 3 (front-left angled)
GET /api/camera/3
Description: Front-left angled camera image, useful to estimate relative lateral offset to the cube.

cURL:
curl -s -H "Accept: image/jpeg" -o cam3.jpg http://127.0.0.1:5000/api/camera/3



## 3) Usage Patterns for RL

Typical RL step loop (high level):
1) Observe:
   - GET /api/joints for proprioception (joint angles/velocities)
   - GET /api/camera/{id} (e.g., 3) for vision (red cube localization via HSV threshold)
2) Decide action:
   - Compute small deltas for joints 0..4 (keep within limits and reasonable step sizes)
   - Convert to absolute targets by adding deltas to current angles; clamp to [lower, upper]
3) Act:
   - POST /api/joints with new absolute positions
4) Reward:
   - Compute pixel distance between end-effector projection and red cube centroid (negative distance)
   - Add shaping terms (e.g., small penalty on joint motion magnitude)
5) Termination:
   - Success if pixel distance under threshold for N consecutive frames
   - Timeout if max steps reached
6) Reset:
   - GET /api/reset/instant to start a new episode (optionally randomize initial pose within safe limits via a few POST /api/joints commands)



## 4) Error Handling & Latency

- Timeouts:
  - Use sensible HTTP timeouts (e.g., 2–5 seconds) and retry logic for images if needed.
- Rate/Throughput:
  - Avoid spamming camera endpoints; 5–20 Hz step rates are typical for training.
- Validation:
  - Always fetch limits from GET /api/joints and clamp commands accordingly.
- Failures:
  - Treat non-200 responses as transient failures; retry limited times; if persistent, end episode and log.



## 5) Example: Red Cube Localization via OpenCV (HSV Threshold)

This is a reference snippet for extracting a red mask and centroid.

import cv2, numpy as np

def red_centroid_bgr(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red can wrap around hue=0; combine two ranges (tune as needed)
    lower1 = np.array([0, 120, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 1e-5:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

Notes:
- You may need to adjust HSV bounds depending on lighting/simulator rendering.
- Optional: use morphology (open/close) to clean noise.



## 6) Safety & Good Practices

- Limit step size per action to respect maxVelocity and ensure stability.
- Smooth actions with low-pass filtering if needed.
- Validate all server responses; handle exceptions to avoid crashing training loops.
- Log all commands and responses during development.
- For repeatability, fix random seeds and initial pose distributions.



## 7) Troubleshooting

- Camera returns empty/invalid images:
  - Verify simulator is running and endpoints reachable.
  - Check image decode logic (ensure jpeg and correct headers).
- Joint POST not moving the robot:
  - Confirm positions are within limits.
  - Ensure using absolute positions, not deltas, in the payload.
- Reset has no effect:
  - Confirm 200 OK on POST /api/reset/instant and wait briefly before new commands.


