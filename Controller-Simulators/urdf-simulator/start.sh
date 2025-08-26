#!/bin/bash

# Script to start the PyBullet URDF Simulator
# This script activates the virtual environment and runs the PybulletViewer
# Supports --loadbalance N option to start N instances with load balancing

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Function to start a single instance
start_instance() {
    local port=$1
    local extra_args=$2

    # Activate virtual environment for this instance
    source .venv/bin/activate

    # Start the instance with specified port and extra arguments
    echo "Starting instance on port $port"
    python scripts/PybulletViewer.py --http-port $port $extra_args &
    echo "Instance on port $port started with PID $!"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install pybullet flask"
    exit 1
fi

# Check if PyBullet is installed
if ! .venv/bin/python -c "import pybullet" 2>/dev/null; then
    echo "Error: PyBullet is not installed in the virtual environment."
    echo "Please install it with: pip install pybullet"
    exit 1
fi

# Parse arguments for load balancing option
LOAD_BALANCE_COUNT=0
EXTRA_ARGS=""

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --loadbalance)
            LOAD_BALANCE_COUNT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ "$LOAD_BALANCE_COUNT" -gt 0 ]; then
    echo "Starting $LOAD_BALANCE_COUNT independent instances..."

    # Start instances on ports 5000 to 5000+LOAD_BALANCE_COUNT-1
    for ((i=0; i<=$LOAD_BALANCE_COUNT; i++)); do
        port=$((5000 + i))
        start_instance $port "--direct $EXTRA_ARGS"
        sleep 1  # Stagger instance starts
    done

    # Wait for all background processes
    echo "All instances started. Press Ctrl+C to stop everything."
    wait

    echo "All instances stopped."
else
    # Normal single instance mode
    echo "Activating virtual environment..."
    source .venv/bin/activate

    # Start the PybulletViewer
    echo "Starting PyBullet URDF Simulator..."
    echo "Press Ctrl+C to stop the simulation"
    echo ""

    python scripts/PybulletViewer.py $EXTRA_ARGS
fi