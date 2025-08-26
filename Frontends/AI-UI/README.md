# Robot Arm AI Chat Interface

This application provides a chat interface that connects Mistral AI with a URDF robot arm simulator. The AI can control the robot arm and observe camera images through tools.

## Features

- Chat interface with Mistral AI
- Tools for controlling robot arm joints
- Camera image observation
- Automatic camera observation for AI self-awareness
- Web-based UI with real-time updates
- Environment variable configuration

## Requirements

- Python 3.7+
- URDF simulator running (from Controller-Simulators/urdf-simulator)
- Mistral API key (optional for full functionality)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file by copying the example:
```bash
cp .env.example .env
```

3. Edit the `.env` file to add your Mistral API key:
```
MISTRAL_API_KEY=your_api_key_here
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file with the following variables:

```
# Mistral API Configuration
MISTRAL_API_KEY=your_api_key_here

# URDF Simulator Configuration
SIMULATOR_HOST=127.0.0.1
SIMULATOR_PORT=5000
```

## Usage

1. Make sure the URDF simulator is running:
```bash
cd Controller-Simulators/urdf-simulator
python scripts/PybulletViewer.py --gui --urdf urdf/five_dof_arm.urdf
```

2. Run the application:
```bash
python main.py
```

3. Open your browser to http://localhost:8000

4. Chat with the AI to control the robot arm and observe camera images

## Tools Available to AI

1. `get_joints` - Get current joint states
2. `set_joint` - Set a specific joint position
3. `set_multiple_joints` - Set multiple joint positions
4. `reset_joints` - Reset all joints to neutral position
5. `read_cam_image` - Read image from a specific camera
6. `self_read_cam_image` - Internal tool for AI to automatically observe cameras

## Project Structure

```
Frontends/AI-UI/
├── main.py                # Main application entry point
├── simulator_client.py   # HTTP client for URDF simulator
├── mistral_client.py     # Mistral API integration with tools
├── chat_ui.py             # FastAPI web interface
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment file
├── templates/
│   └── chat.html          # HTML template
├── static/
│   └── style.css          # CSS styles
└── README.md              # This file
```

## Development

To run tests:
```bash
python test_integration.py
```

## Notes

- The application automatically observes camera images at regular intervals
- Camera images are displayed in the UI and also sent to the AI for observation
- The AI can use tools to control the robot arm and read camera images