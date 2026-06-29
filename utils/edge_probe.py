#!/usr/bin/env python3
"""Minimal edge-device RAM probe.

Run as a subprocess — intentionally imports ONLY onnxruntime, numpy, cv2.
No insightface, no scipy, no webcam, no Anaconda extras.
Measures the true model+inference RSS that an edge device would pay.

Prints one line to stdout:
    <rss_after_load_MB> <rss_after_warmup_MB>
then exits.
"""
import sys
import numpy as np
import cv2
import onnxruntime as ort
import psutil

def _rss():
    return psutil.Process().memory_info().rss / 1e6

det_path, rec_path = sys.argv[1], sys.argv[2]

# Edge-tuned session options (same as make_edge_session_options)
def _opts():
    o = ort.SessionOptions()
    o.intra_op_num_threads     = 2
    o.inter_op_num_threads     = 1
    o.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    o.enable_mem_pattern       = False
    o.add_session_config_entry("session.arena_extend_strategy", "kSameAsRequested")
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return o

def _providers():
    available = ort.get_available_providers()
    if "XNNPACKExecutionProvider" in available:
        return ["XNNPACKExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

_ep = _providers()
det_sess = ort.InferenceSession(det_path, sess_options=_opts(), providers=_ep)
rec_sess = ort.InferenceSession(rec_path, sess_options=_opts(), providers=_ep)
print(f"EP: {det_sess.get_providers()[0]}", file=__import__('sys').stderr)
rss_load = _rss()

det_in = det_sess.get_inputs()[0].name
rec_in = rec_sess.get_inputs()[0].name

# Warm up: realistic inputs (320×320 det, 112×112 rec)
det_blob = np.random.randn(1, 3, 320, 320).astype(np.float32)
rec_blob = np.random.randn(1, 3, 112, 112).astype(np.float32)
for _ in range(10):
    det_sess.run(None, {det_in: det_blob})
    rec_sess.run(None, {rec_in: rec_blob})

rss_warm = _rss()

print(f"{rss_load:.1f} {rss_warm:.1f}", flush=True)
