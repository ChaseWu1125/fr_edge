from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FaceDetection:
    """Single-face detection result."""
    bbox:       np.ndarray   # shape (4,)  — [x1, y1, x2, y2]
    landmarks:  np.ndarray   # shape (5,2) — 5 facial keypoints (x, y)
    confidence: float


class BaseDetector(ABC):
    """Interface for face detection.

    Implement this to plug in any detector backend:
    SCRFD, RetinaFace, YOLOv8-face, or a future ONNX/TRT version.
    """

    @abstractmethod
    def load(self) -> None:
        """Initialise and load model weights into memory."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in a BGR frame.

        Args:
            frame: HxWx3 uint8 BGR image (OpenCV native format).
        Returns:
            Possibly-empty list of FaceDetection results.
        """
        ...
