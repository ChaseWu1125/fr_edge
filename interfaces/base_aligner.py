from abc import ABC, abstractmethod

import numpy as np


class BaseAligner(ABC):
    """Interface for face alignment / cropping.

    Consumes a full BGR frame + 5 landmarks and produces a canonical
    112×112 BGR crop ready for feature extraction.
    """

    @abstractmethod
    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Warp and crop a face to a 112×112 canonical patch.

        Args:
            frame:     HxWx3 uint8 BGR image.
            landmarks: (5, 2) float32 array of (x, y) keypoint coords.
        Returns:
            112×112×3 uint8 BGR face image.
        """
        ...
