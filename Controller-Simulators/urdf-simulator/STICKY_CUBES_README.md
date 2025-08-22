# Sticky Cube System for Robot Arm Simulation

## Overview

The Sticky Cube System enables cubes to automatically "stick" to the robot's end effector when touched and be released when placed into containers. This creates an intuitive pick-and-place interaction without requiring complex gripper mechanics.

## How It Works

### 1. 🎮 Wrist Roll Control
- **Gripper Activation**: Rotate wrist roll joint to **< -0.5 radians** to activate gripper
- **Gripper Release**: Rotate wrist roll joint to **> 0.5 radians** to release gripper
- The wrist roll joint acts as a virtual gripper control mechanism

### 2. 🎯 Proximity Detection
- When gripper is active, system monitors distance between end effector and cubes
- Attachment occurs when end effector is within **8cm** of a cube and gripper is active

### 3. 🔗 Physics-Based Attachment
- Uses PyBullet's `createConstraint` with `JOINT_FIXED` type
- Creates a rigid physics connection between the end effector and the cube
- The cube becomes part of the robot's kinematic chain temporarily

### 4. 🚚 Transport Phase
- The attached cube follows all end effector movements
- Physics simulation handles collision detection and dynamics
- Only one cube can be attached at a time

### 5. 📦 Release Mechanisms
- **Manual Release**: Rotate wrist roll > 0.5 radians to release cube
- **Auto Release**: System automatically releases when cube enters container area
- Cube falls under gravity after constraint removal

## Usage

### Basic Simulation
```bash
# Activate virtual environment
source .venv/bin/activate

# Run with sticky cubes enabled (default)
python scripts/PybulletViewer.py --cube-count 3 --container-count 2

# Run without sticky cubes
python scripts/PybulletViewer.py --disable-sticky-cubes
```

### Interactive Demo
```bash
# Run the automated demonstration
python scripts/sticky_cube_demo.py
```

### Command Line Options
- `--cube-count N`: Number of cubes to spawn (default: 3)
- `--container-count N`: Number of containers to spawn (default: 2)
- `--enable-sticky-cubes`: Enable sticky cube system (default)
- `--disable-sticky-cubes`: Disable sticky cube system

## Manual Control

You can control the robot manually using:

1. **GUI Sliders**: Use the PyBullet GUI sliders to move joints
2. **HTTP API**: Send joint commands via HTTP (if enabled)
3. **Keyboard/Mouse**: Use PyBullet's built-in camera controls

### HTTP API Examples
```bash
# Move shoulder yaw joint to 0.5 radians
curl -X GET "http://localhost:5000/api/joint/0/0.5"

# Get current joint states
curl -X GET "http://localhost:5000/api/joints"
```

## Configuration

### Sticky System Parameters
You can modify these in `sticky_cube_system.py`:

```python
# Distance threshold for attachment (meters)
self.attachment_distance_threshold = 0.05  # 5cm

# Physics constraint force
self.constraint_force = 500.0

# Container detection threshold (meters)  
self.container_detection_threshold = 0.1  # 10cm
```

### Robot Configuration
- End effector is automatically detected (usually the last link)
- Works with any URDF robot that has an identifiable end effector
- Default robot: 5-DOF arm (`urdf/five_dof_arm.urdf`)

## Troubleshooting

### Cubes Not Sticking
- **Check distance**: End effector must be within 5cm of cube center
- **Check attachment status**: Only one cube can be attached at a time
- **Verify end effector**: System auto-detects end effector link

### Cubes Not Releasing
- **Check container position**: Cube must be within container boundaries
- **Verify container registration**: Containers must be registered with system
- **Check detection threshold**: May need to get closer to container center

### Performance Issues
- **Reduce cube count**: Use `--cube-count 1` for testing
- **Disable real-time**: Remove `--realtime` flag for faster simulation
- **Use headless mode**: Add `--direct` flag to disable GUI

## Technical Details

### Architecture
```
PybulletViewer.py          # Main simulation loop
├── sticky_cube_system.py  # Core sticky cube logic
├── world_helpers.py       # World setup (cubes, containers)
└── http_api.py            # HTTP control interface
```

### Key Classes
- `StickyCubeSystem`: Main system controller
- `CubeState`: Tracks individual cube states
- `Container`: Represents container objects

### Physics Implementation
- Uses PyBullet's constraint system for attachment
- Maintains physics accuracy during transport
- Handles collision detection automatically

## Examples

### Simple Pick and Place
1. Start simulation: `python scripts/PybulletViewer.py`
2. Use GUI sliders to move robot arm near a cube
3. Watch cube automatically attach when close enough
4. Move arm to container area
5. Watch cube automatically release into container

### Automated Demo
1. Run: `python scripts/sticky_cube_demo.py`
2. Watch automated sequence of pick and place operations
3. Observe attachment/release behavior

## Future Enhancements

Potential improvements:
- Multiple cube attachment (gripper simulation)
- Force-based attachment (pressure sensitivity)
- Visual/audio feedback for attachment events
- Container-specific cube sorting
- Collision avoidance during transport