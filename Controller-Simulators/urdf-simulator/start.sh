#!/bin/bash

# Script to start the PyBullet URDF Simulator
# This script activates the virtual environment and runs the PybulletViewer

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install pybullet flask"
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if PyBullet is installed
if ! python -c "import pybullet" 2>/dev/null; then
    echo "Error: PyBullet is not installed in the virtual environment."
    echo "Please install it with: pip install pybullet"
    exit 1
fi

# Start the PybulletViewer
echo "Starting PyBullet URDF Simulator..."
echo "Press Ctrl+C to stop the simulation"
echo ""

python scripts/PybulletViewer.py "$@"