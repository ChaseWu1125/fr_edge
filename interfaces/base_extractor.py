from abc import ABC, abstractmethod

import numpy as np


class BaseExtractor(ABC):
    """Interface for facial feature extraction.

    Consumes a 112×112 aligned face image and produces an L2-normalised
    512-D embedding vector.

    Swap hook: subclass this and inject into RealtimePipeline to switch
    from MobileFaceNet → GhostFaceNet, or prototype → ONNX/INT8 version.
    """

    @abstractmethod
    def load(self) -> None:
        """Initialise and load model weights into memory."""
        ...

    @abstractmethod
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """Extract a 512-D embedding from a 112×112 face image.

        Args:
            face_img: 112×112×3 uint8 BGR face crop (aligned).
        Returns:
            (512,) float32 L2-normalised embedding.
        """
        ...
