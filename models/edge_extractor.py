"""Edge MobileFaceNet extractor — pure onnxruntime, no insightface dependency."""

import cv2
import numpy as np
import onnxruntime as ort

import config
from interfaces.base_extractor import BaseExtractor


class EdgeExtractor(BaseExtractor):
    """MobileFaceNet feature extractor for edge deployment — pure ORT."""

    def __init__(self):
        self._session    = None
        self._input_name = "input.1"

    def load(self) -> None:
        model_path = config.REC_INT8_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"INT8 extractor not found: {model_path}\n"
                "  Run: python scripts/quantize_models.py --dynamic"
            )
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=config.make_edge_session_options(),
            providers=["CPUExecutionProvider"],
        )
        print("[EdgeExtractor] loaded INT8 MobileFaceNet")

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Args:
            face_img: 112×112×3 uint8 BGR face crop.
        Returns:
            (512,) float32 L2-normalised embedding.
        """
        if self._session is None:
            raise RuntimeError("Call load() before extract().")
        blob = cv2.dnn.blobFromImages(
            [face_img], 1.0 / 127.5, (112, 112),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        emb = self._session.run(None, {self._input_name: blob})[0].flatten()
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)
