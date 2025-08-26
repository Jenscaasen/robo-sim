#!/usr/bin/env bash
set -euo pipefail

# Create and populate a local virtual environment for the positioning project.
# Usage:
#   bash Reinforcement_learning/positioning/setup_venv.sh
# Activate:
#   source Reinforcement_learning/positioning/.venv/bin/activate
# Train:
#   python Reinforcement_learning/positioning/TrainPositioner.py --tensorboard

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements.txt not found at: ${REQ_FILE}" >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

echo
echo "Virtual environment ready."
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Then start training, e.g.:"
echo "  python Reinforcement_learning/positioning/TrainPositioner.py --tensorboard"