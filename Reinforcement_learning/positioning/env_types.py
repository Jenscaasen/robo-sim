from dataclasses import dataclass
from typing import Tuple
from RedDetector import Detection  # local import


@dataclass
class ViewDetections:
    red: Detection
    purple: Detection
    size: Tuple[int, int]  # (width, height)