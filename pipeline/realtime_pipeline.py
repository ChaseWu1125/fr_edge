import time

import cv2
import numpy as np

import config
from interfaces.base_aligner import BaseAligner
from interfaces.base_detector import BaseDetector
from interfaces.base_extractor import BaseExtractor
from utils.face_database import FaceDatabase
from utils.visualizer import draw_detection, draw_fps, draw_status_bar


class RealtimePipeline:
    """Streaming face-recognition controller.

    Wires a detector, aligner, extractor, and face database together into
    a frame-by-frame identification loop driven by a webcam.

    All four components are injected, so any can be swapped independently:
      • detector  — SCRFD → YOLOv8-face, ONNX export, CoreML, …
      • aligner   — norm_crop → custom warp, TRT accelerated, …
      • extractor — MobileFaceNet → GhostFaceNet, INT8 ONNX, …
      • database  — pickle → Redis, SQLite, Faiss ANN index, …
    """

    def __init__(
        self,
        detector:       BaseDetector,
        aligner:        BaseAligner,
        extractor:      BaseExtractor,
        database:       FaceDatabase,
        webcam_index:   int   = config.WEBCAM_INDEX,
        threshold:      float = config.RECOGNITION_THRESHOLD,
        show_landmarks: bool  = True,
    ):
        self._detector       = detector
        self._aligner        = aligner
        self._extractor      = extractor
        self._db             = database
        self._webcam_index   = webcam_index
        self._threshold      = threshold
        self._show_landmarks = show_landmarks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, cap: cv2.VideoCapture = None) -> None:
        """Open webcam and run recognition loop until user presses 'q'.

        Pass a pre-opened VideoCapture to skip _open_capture() — useful when
        the caller needs to measure the camera's RSS cost before inference starts.
        """
        owns_cap = cap is None
        if owns_cap:
            cap = self._open_capture()
        try:
            self._loop(cap)
        finally:
            if owns_cap:
                cap.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._webcam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {self._webcam_index}.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # keep only latest frame; avoids queued-frame RAM
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _loop(self, cap: cv2.VideoCapture) -> None:
        prev_time = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Pipeline] Frame capture failed — retrying…")
                continue

            self._process_frame(frame)

            now = time.perf_counter()
            draw_fps(frame, 1.0 / max(now - prev_time, 1e-6))
            draw_status_bar(frame, len(self._db))
            prev_time = now

            cv2.imshow("Face Recognition — FR_Project", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def _process_frame(self, frame: np.ndarray) -> None:
        detections = self._detector.detect(frame)

        for det in detections:
            try:
                aligned   = self._aligner.align(frame, det.landmarks)
                embedding = self._extractor.extract(aligned)
                name, sim = self._db.identify(embedding, self._threshold)
            except Exception as exc:
                print(f"[Pipeline] Per-face processing error: {exc}")
                name, sim = None, 0.0

            draw_detection(
                frame,
                det.bbox,
                name,
                sim,
                landmarks=det.landmarks if self._show_landmarks else None,
            )
