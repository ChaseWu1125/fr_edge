from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH  = DATA_DIR / "face_database.pkl"

# Detection
DET_THRESHOLD = 0.5

# Recognition
RECOGNITION_THRESHOLD = 0.40  # cosine similarity; raise to 0.45 for stricter matching

# INT8 quantized model paths
QUANTIZED_DIR = DATA_DIR / "quantized"
DET_INT8_PATH = QUANTIZED_DIR / "det_500m_int8.onnx"
REC_INT8_PATH = QUANTIZED_DIR / "w600k_mbf_int8.onnx"

# Webcam / display
WEBCAM_INDEX = 0       # change to match your device (e.g. /dev/video0 = 0)
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Edge-device deployment
EDGE_DET_SIZE    = (320, 320)
EDGE_NUM_THREADS = 2   # set to the core count of your target SoC


def make_edge_session_options():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = EDGE_NUM_THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern   = False
    opts.add_session_config_entry("session.arena_extend_strategy", "kSameAsRequested")
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level   = 3
    return opts
