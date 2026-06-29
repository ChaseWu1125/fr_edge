#!/usr/bin/env python3
"""Register faces into the local database using edge components (pure ORT).

Usage:
    # 3 webcam snapshots (default)
    python scripts/register_face.py --name "Alice"

    # More samples for better accuracy
    python scripts/register_face.py --name "Bob" --samples 5

    # From a static photo
    python scripts/register_face.py --name "Carol" --image path/to/photo.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.edge_scrfd_detector import EdgeSCRFDDetector
from models.edge_aligner import EdgeAligner
from models.edge_extractor import EdgeExtractor
from utils.face_database import FaceDatabase


def _build_components():
    detector  = EdgeSCRFDDetector()
    aligner   = EdgeAligner()
    extractor = EdgeExtractor()
    detector.load()
    extractor.load()
    return detector, aligner, extractor, FaceDatabase()


def register_from_image(name: str, image_path: Path) -> None:
    detector, aligner, extractor, db = _build_components()

    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"[Error] Cannot read image: {image_path}")

    detections = detector.detect(frame)
    if not detections:
        sys.exit("[Error] No face detected in the image.")
    if len(detections) > 1:
        print(f"[Warning] {len(detections)} faces found — using highest-confidence one.")

    det       = max(detections, key=lambda d: d.confidence)
    aligned   = aligner.align(frame, det.landmarks)
    embedding = extractor.extract(aligned)
    db.add(name, embedding)
    print(f"[OK] Registered '{name}' from {image_path}.")


def register_from_webcam(name: str, num_samples: int) -> None:
    detector, aligner, extractor, db = _build_components()

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        sys.exit("[Error] Cannot open webcam.")

    print(f"[Info] Capturing {num_samples} sample(s) for '{name}'.")
    print("       Press SPACE to capture a sample, Q to quit early.\n")

    count = 0
    while count < num_samples:
        ok, frame = cap.read()
        if not ok:
            continue

        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2 = det.bbox[:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

        status = f"[{count}/{num_samples}] SPACE=capture  Q=quit"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and detections:
            det       = max(detections, key=lambda d: d.confidence)
            aligned   = aligner.align(frame, det.landmarks)
            embedding = extractor.extract(aligned)
            db.add(name, embedding)
            count += 1
            print(f"  [{count}/{num_samples}] Sample captured.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if count > 0:
        print(f"\n[OK] Registered {count} sample(s) for '{name}'.")
    else:
        print("\n[Warning] No samples captured.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a face into the FR_Project database.")
    ap.add_argument("--name",    required=True,      help="Identity label, e.g. 'Alice'")
    ap.add_argument("--image",   default=None,        help="Path to a static photo (skips webcam)")
    ap.add_argument("--samples", type=int, default=3, help="Number of webcam snapshots (default: 3)")
    args = ap.parse_args()

    if args.image:
        register_from_image(args.name, Path(args.image))
    else:
        register_from_webcam(args.name, args.samples)


if __name__ == "__main__":
    main()
