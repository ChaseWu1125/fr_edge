# FR_Project — Edge Deployment

Real-time face recognition pipeline for Linux edge devices.
Pure ORT inference — no InsightFace, no scipy, no Anaconda dependencies.

**Memory:** ~75–100 MB RSS on device (measured ~101 MB on macOS probe; Linux will be lower).  
**Models:** SCRFD-500M detector + MobileFaceNet extractor, both static INT8 ONNX.  
**Detection grid:** 320×320 — suitable for faces ≥ 30 px; adjust `EDGE_DET_SIZE` in `config.py` if needed.

---

## Requirements

- Python 3.9+
- A camera accessible via `cv2.VideoCapture` (V4L2 on Linux)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.py` before running:

- **`WEBCAM_INDEX`** — set to your camera index (default `0` = `/dev/video0`)
- **`EDGE_NUM_THREADS`** — set to the core count of your SoC (default `2`)
- **`RECOGNITION_THRESHOLD`** — cosine similarity cutoff (default `0.40`; raise to `0.45` for stricter matching)

## Register Faces

Run this on any machine that has the same models (or on the device itself):

```bash
# From webcam (3 snapshots by default)
python scripts/register_face.py --name "Alice"

# More samples for better accuracy
python scripts/register_face.py --name "Alice" --samples 5

# From a static photo
python scripts/register_face.py --name "Alice" --image path/to/photo.jpg
```

Registered faces are saved to `data/face_database.pkl`.  
The database included in this package already contains 3 identities — remove or replace it if deploying fresh.

## Run

```bash
# Standard
python main.py

# With live RAM stats (prints RSS every 2s)
python main.py --monitor

# Adjust monitor interval
python main.py --monitor --interval 5
```

Press **Q** to quit.

## Measure Device Memory (no camera needed)

```bash
python utils/edge_probe.py data/quantized/det_500m_int8.onnx data/quantized/w600k_mbf_int8.onnx
```

Prints `<rss_after_load_MB> <rss_after_warmup_MB>` — the true model+inference cost with no camera or display overhead.

## File Structure

```
fr_edge/
├── main.py                        # Entry point
├── config.py                      # All tunable parameters
├── requirements.txt
├── interfaces/                    # Abstract base classes
├── models/
│   ├── edge_scrfd_detector.py     # SCRFD face detector (pure ORT)
│   ├── edge_aligner.py            # Face alignment (cv2 only)
│   └── edge_extractor.py          # MobileFaceNet extractor (pure ORT)
├── pipeline/
│   └── realtime_pipeline.py       # cap.read → detect → align → extract → DB
├── utils/
│   ├── face_database.py           # Pickle-backed identity store
│   ├── visualizer.py              # OpenCV overlay drawing
│   └── edge_probe.py              # Standalone RAM measurement tool
├── scripts/
│   └── register_face.py           # Face enrollment script
└── data/
    ├── quantized/
    │   ├── det_500m_int8.onnx     # SCRFD INT8 detector (0.7 MB)
    │   └── w600k_mbf_int8.onnx   # MobileFaceNet INT8 extractor (3.5 MB)
    └── face_database.pkl          # Registered identities
```

## Known Limitations

- Detection misses faces smaller than ~30 px at 320×320 grid. Increase `EDGE_DET_SIZE` to `(640, 640)` if needed (higher RAM cost).
- `register_face.py` requires a display for webcam mode. Use `--image` flag on headless devices.
- NPU execution provider (NNAPI, QNN, etc.) not wired — CPU only. To enable, add the appropriate EP to `providers=` in `make_edge_session_options`.
