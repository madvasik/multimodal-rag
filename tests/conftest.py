"""Stabilize native libs (OpenMP) when tests mix torch/faiss."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
