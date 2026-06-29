#!/usr/bin/env python3
"""FR_Project — edge deployment entry point.

Runs the real-time face recognition pipeline using pure ORT (no InsightFace).
INT8 ONNX models must exist in data/quantized/ before running.

Usage:
    python main.py
    python main.py --monitor
    python main.py --interval 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from models.edge_scrfd_detector import EdgeSCRFDDetector
from models.edge_aligner import EdgeAligner
from models.edge_extractor import EdgeExtractor
from pipeline.realtime_pipeline import RealtimePipeline
from utils.face_database import FaceDatabase


def main() -> None:
    ap = argparse.ArgumentParser(description="FR_Project edge demo")
    ap.add_argument("--monitor",  action="store_true", help="Print live RAM stats")
    ap.add_argument("--interval", type=float, default=2.0, help="Monitor interval in seconds")
    args = ap.parse_args()

    print("[Boot] Mode: EDGE  (pure ORT, INT8 models, 320×320 detection)")

    monitor = None
    if args.monitor:
        from utils.memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(interval=args.interval)

    print("[Boot] Loading models…")
    detector  = EdgeSCRFDDetector()
    aligner   = EdgeAligner()
    extractor = EdgeExtractor()
    database  = FaceDatabase()

    detector.load()
    extractor.load()

    n = len(database)
    if n == 0:
        print("\n[Warning] Face database is empty.\n"
              "          Run: python scripts/register_face.py --name 'Your Name'\n")
    else:
        print(f"[Boot] Database: {n} registered identit{'y' if n == 1 else 'ies'}.")

    print("[Boot] Starting webcam — press Q to quit.\n")

    import cv2
    import psutil
    _proc = psutil.Process()
    _rss_before_cam = _proc.memory_info().rss / 1e6
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    _cam_overhead = _proc.memory_info().rss / 1e6 - _rss_before_cam

    if monitor:
        monitor.exclude_overhead(_cam_overhead)
        monitor.start()

    pipeline = RealtimePipeline(
        detector=detector,
        aligner=aligner,
        extractor=extractor,
        database=database,
    )

    try:
        pipeline.run(cap=cap)
    finally:
        cap.release()
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main()
