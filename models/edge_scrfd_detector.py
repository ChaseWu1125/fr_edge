"""Edge SCRFD detector — zero insightface dependency at inference time.

Loads det_500m_int8.onnx directly via onnxruntime and reimplements SCRFD
postprocessing (anchor generation, distance→bbox/kps decoding, NMS) so that
scipy / albumentations / matplotlib are never imported on the edge device.

Uses EDGE_DET_SIZE (320×320) by default — cuts ORT activation arena ~4× vs
the desktop 640×640 grid.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

import config
from interfaces.base_detector import BaseDetector, FaceDetection

_STRIDES     = [8, 16, 32]
_NUM_ANCHORS = 2   # SCRFD-500M: 2 anchors per spatial location


# ── SCRFD math helpers ────────────────────────────────────────────────────────

def _anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """(H*W*num_anchors, 2) anchor center pixel coords for one feature-map level."""
    cy, cx = np.mgrid[:height, :width]
    centers = np.stack([cx, cy], axis=-1).astype(np.float32).reshape(-1, 2) * stride
    # Each spatial position has _NUM_ANCHORS anchors at the same centre
    return np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)


def _distance2bbox(centers: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Decode distance predictions → [x1, y1, x2, y2] boxes."""
    return np.stack([
        centers[:, 0] - preds[:, 0],
        centers[:, 1] - preds[:, 1],
        centers[:, 0] + preds[:, 2],
        centers[:, 1] + preds[:, 3],
    ], axis=-1)


def _distance2kps(centers: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Decode distance predictions → (N, 5, 2) keypoints."""
    pts = []
    for i in range(0, preds.shape[1], 2):
        pts.append(centers[:, 0] + preds[:, i])
        pts.append(centers[:, 1] + preds[:, i + 1])
    return np.stack(pts, axis=-1).reshape(-1, 5, 2)


def _nms(bboxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> np.ndarray:
    """NMS via cv2 — no scipy required."""
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.0, nms_thresh)
    return np.array(keep).flatten() if len(keep) else np.array([], dtype=np.int32)


# ── Detector ─────────────────────────────────────────────────────────────────

class EdgeSCRFDDetector(BaseDetector):
    """SCRFD face detector for edge deployment — pure ORT, no insightface."""

    def __init__(
        self,
        det_size:      tuple = config.EDGE_DET_SIZE,
        det_threshold: float = config.DET_THRESHOLD,
        nms_threshold: float = 0.4,
    ):
        self._det_size      = det_size          # (W, H) passed to session
        self._det_threshold = det_threshold
        self._nms_threshold = nms_threshold
        self._session       = None
        self._center_cache: dict = {}

    def load(self) -> None:
        model_path = config.DET_INT8_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"INT8 detector not found: {model_path}\n"
                "  Run: python scripts/quantize_models.py --dynamic"
            )
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=config.make_edge_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        # Group 9 outputs by last dim: 1=scores, 4=bboxes, 10=kps; sort by size desc
        self._score_names, self._bbox_names, self._kps_names = self._group_outputs()
        print(f"[EdgeSCRFDDetector] loaded INT8  det_size={self._det_size}")

    def _group_outputs(self):
        scores, bboxes, kpss = [], [], []
        for out in self._session.get_outputs():
            last = out.shape[-1]
            size = out.shape[0] if isinstance(out.shape[0], int) else 0
            entry = (size, out.name)
            if last == 1:
                scores.append(entry)
            elif last == 4:
                bboxes.append(entry)
            elif last == 10:
                kpss.append(entry)
        key = lambda x: -x[0]   # sort descending by spatial size → stride ascending
        return (
            [n for _, n in sorted(scores, key=key)],
            [n for _, n in sorted(bboxes, key=key)],
            [n for _, n in sorted(kpss,   key=key)],
        )

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        if self._session is None:
            raise RuntimeError("Call load() before detect().")

        det_img, scale = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0,
            (det_img.shape[1], det_img.shape[0]),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        raw = self._session.run(self._output_names, {self._input_name: blob})
        output_map = dict(zip(self._output_names, raw))

        input_h, input_w = det_img.shape[:2]

        all_scores, all_bboxes, all_kpss = [], [], []

        for i, stride in enumerate(_STRIDES):
            sh = input_h // stride
            sw = input_w // stride

            scores = output_map[self._score_names[i]].flatten()
            bbox_p = output_map[self._bbox_names[i]] * stride
            kps_p  = output_map[self._kps_names[i]]  * stride

            key = (sh, sw, stride)
            if key not in self._center_cache:
                self._center_cache[key] = _anchor_centers(sh, sw, stride)
            centers = self._center_cache[key]

            mask = scores >= self._det_threshold
            if not mask.any():
                continue

            all_scores.append(scores[mask])
            all_bboxes.append(_distance2bbox(centers[mask], bbox_p[mask]))
            all_kpss.append(_distance2kps(centers[mask], kps_p[mask]))

        if not all_scores:
            return []

        scores  = np.concatenate(all_scores)
        bboxes  = np.concatenate(all_bboxes)
        kpss    = np.concatenate(all_kpss)

        keep = _nms(bboxes, scores, self._nms_threshold)
        if len(keep) == 0:
            return []

        bboxes = bboxes[keep] / scale
        kpss   = kpss[keep]   / scale
        scores = scores[keep]

        return [
            FaceDetection(
                bbox=bboxes[i].astype(np.int32),
                landmarks=kpss[i].astype(np.float32),
                confidence=float(scores[i]),
            )
            for i in range(len(keep))
        ]

    def _letterbox(self, img: np.ndarray):
        """Fit img into det_size while keeping aspect ratio; return (padded, scale)."""
        target_w, target_h = self._det_size
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, scale
