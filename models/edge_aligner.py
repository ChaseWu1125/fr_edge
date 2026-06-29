"""Edge face aligner — cv2 only, no scikit-image / scipy dependency."""

import cv2
import numpy as np

from interfaces.base_aligner import BaseAligner

# Canonical 5-point ArcFace template for 112×112 output
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class EdgeAligner(BaseAligner):
    """Similarity-transform face alignment using cv2 only (no scikit-image)."""

    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Args:
            frame:     BGR image (any size).
            landmarks: (5, 2) float32 facial keypoints from the detector.
        Returns:
            (112, 112, 3) uint8 aligned face crop.
        """
        M, _ = cv2.estimateAffinePartial2D(
            landmarks, _ARCFACE_DST, method=cv2.LMEDS,
        )
        return cv2.warpAffine(frame, M, (112, 112), borderValue=0)
