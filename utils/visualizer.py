from typing import Optional

import cv2
import numpy as np

_GREEN  = (0, 220,   0)
_RED    = (0,   0, 220)
_YELLOW = (0, 220, 220)
_WHITE  = (255, 255, 255)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection(
    frame:      np.ndarray,
    bbox:       np.ndarray,
    name:       Optional[str],
    similarity: float,
    landmarks:  Optional[np.ndarray] = None,
) -> None:
    """Overlay bounding box, label, and optional landmarks on *frame* in-place."""
    x1, y1, x2, y2 = bbox[:4].astype(int)
    color = _GREEN if name else _RED
    label = f"{name} ({similarity:.2f})" if name else f"Unknown ({similarity:.2f})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)

    if landmarks is not None:
        for (lx, ly) in landmarks.astype(int):
            cv2.circle(frame, (lx, ly), 2, _YELLOW, -1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), _FONT, 0.8, _GREEN, 2, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, num_identities: int) -> None:
    h = frame.shape[0]
    text = f"DB: {num_identities} identit{'y' if num_identities == 1 else 'ies'}  |  Q = quit"
    cv2.putText(frame, text, (10, h - 10), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
