# Dataset Compiler

A Python-based system for compiling robot dataset from folders containing joint position data (JSON) and camera images.

## Features

- Extracts joint positions from JSON files (shoulder_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll)
- Detects black blob (end-effector) tip positions in camera images using OpenCV
- Finds the tip point furthest from the yellow link for accurate positioning
- Generates CSV files with semicolon-separated values
- Interactive folder selection via bash script

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy
```

## Usage

### Interactive Mode (Recommended)
```bash
./run.sh
```
This will:
- Activate the virtual environment
- List available dataset folders
- Allow you to select which folder to process
- Generate the corresponding CSV file

### Direct Mode
```bash
source venv/bin/activate
python dataset_compiler.py <folder_name>
```

### Simple Wrapper
```bash
source venv/bin/activate
python compile_dataset.py <folder_name>
```

## Input Format

### Folder Structure
```
dataset_YYYYMMDD_HHMMSS/
├── datarow-00001.json
├── datarow-00001-cam1.jpg
├── datarow-00001-cam2.jpg
├── datarow-00001-cam3.jpg
├── datarow-00002.json
├── datarow-00002-cam1.jpg
├── datarow-00002-cam2.jpg
├── datarow-00002-cam3.jpg
└── ...
```

### JSON Format
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
  ...
]
```

## Output Format

CSV file with semicolon-separated values:
```
shoulder_yaw;shoulder_pitch;elbow_pitch;wrist_pitch;wrist_roll;cam1_tip_x;cam1_tip_y;cam2_tip_x;cam2_tip_y;cam3_tip_x;cam3_tip_y
0.514462285827634;-0.5487437915021836;0.2427464650363842;-0.9280658443001907;-0.038948032762304186;602;312;207;182;472;224
```

## Files

- `dataset_compiler.py` - Main compilation script with OpenCV blob detection
- `compile_dataset.py` - Simple wrapper script
- `run.sh` - Interactive bash script with folder selection
- `black_blob_detector.py` - Standalone blob detection with visualization
- `simple_black_blob_detector.py` - Minimal blob detection script
- `enhanced_black_blob_detector.py` - Advanced blob detection with tip finding

## Algorithm

The black blob tip detection works by:
1. Converting image to grayscale
2. Creating binary mask for black regions (threshold < 30)
3. Finding contours of black regions
4. Detecting yellow region as reference point
5. Finding the point on the black blob contour furthest from the yellow region
6. Returning pixel coordinates (x, y) of the tip