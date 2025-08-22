# Dataset Creator

A .NET console application that generates robotic arm dataset entries by connecting to a PyBullet HTTP API.

## Features

- Connects to PyBullet HTTP API to control a 5-DOF robotic arm
- Generates random joint configurations within joint limits
- Captures images from 3 different camera angles
- Saves joint configurations as JSON files
- Creates timestamped dataset folders
- Proper error handling and progress reporting

## Prerequisites

- .NET 9.0 or later
- PyBullet viewer running with HTTP API enabled

## Starting the PyBullet Viewer

Before running the dataset creator, start the PyBullet viewer with HTTP API:

```bash
.venv/bin/python scripts/PybulletViewer.py --gui --urdf urdf/five_dof_arm.urdf
```

The viewer will start on `http://127.0.0.1:5000` by default.

## Usage

```bash
dotnet run <number_of_dataset_entries>
```

### Examples

Create 10 dataset entries:
```bash
dotnet run 10
```

Create 100 dataset entries:
```bash
dotnet run 100
```

## Output Structure

The application creates a timestamped folder in the `datasets` directory:

```
datasets/
└── dataset_20250821_115559/
    ├── datarow-00001-cam1.jpg    # Camera 1 image
    ├── datarow-00001-cam2.jpg    # Camera 2 image  
    ├── datarow-00001-cam3.jpg    # Camera 3 image
    ├── datarow-00001.json        # Joint configuration
    ├── datarow-00002-cam1.jpg
    ├── datarow-00002-cam2.jpg
    ├── datarow-00002-cam3.jpg
    ├── datarow-00002.json
    └── ...
```

## Joint Configuration Format

Each JSON file contains the joint names and their positions:

```json
[
  {
    "name": "shoulder_yaw",
    "position": 0.514462285827634
  },
  {
    "name": "shoulder_pitch", 
    "position": -0.5487437915021836
  },
  {
    "name": "elbow_pitch",
    "position": 0.2427464650363842
  },
  {
    "name": "wrist_pitch",
    "position": -0.9280658443001907
  },
  {
    "name": "wrist_roll",
    "position": -0.038948032762304186
  }
]
```

## API Endpoints Used

- `GET /api/joints` - Retrieve joint information and limits
- `GET /api/joint/{id}/{position}/instant` - Set joint position instantly
- `GET /api/camera/{id}` - Capture image from camera (1, 2, or 3)

## Error Handling

The application includes comprehensive error handling:
- Connection failures to the PyBullet API
- Joint setting failures
- Camera capture failures
- File system errors

If any step fails, the application will report the error and continue with the next dataset entry when possible.

## Building

```bash
dotnet build
```

## Running

```bash
dotnet run <number_of_entries>