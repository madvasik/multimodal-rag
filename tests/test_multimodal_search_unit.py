"""Unit tests for pure helpers in src.retrieval.multimodal_search."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# Heavy import once per module (torch, transformers deps).
from src.retrieval import multimodal_search as ms


def test_sorted_embedding_shard_paths_order(tmp_path: Path) -> None:
    d = tmp_path
    (d / "embeddings_00000500.pt").write_bytes(b"x")
    (d / "embeddings_00000000.pt").write_bytes(b"x")
    (d / "other.pt").write_bytes(b"x")
    paths = ms._sorted_embedding_shard_paths(str(d))
    basenames = [os.path.basename(p) for p in paths]
    assert basenames == ["embeddings_00000000.pt", "embeddings_00000500.pt"]


def test_load_shard_rows_tensor_2d(tmp_path: Path) -> None:
    t = torch.randn(3, 4)
    p = tmp_path / "shard.pt"
    torch.save(t, p)
    rows = ms._load_shard_rows(str(p))
    assert len(rows) == 3
    assert all(r.shape == (4,) for r in rows)


def test_load_shard_rows_list_preserved(tmp_path: Path) -> None:
    chunks = [torch.randn(1, 2), torch.randn(1, 2)]
    p = tmp_path / "list.pt"
    torch.save(chunks, p)
    rows = ms._load_shard_rows(str(p))
    assert len(rows) == 2


def test_rows_for_cat_stacks_to_batch() -> None:
    """Vectors must share trailing dims for torch.cat(dim=0)."""
    a = torch.randn(1, 5)
    b = torch.randn(5)
    c = torch.randn(5)
    out = ms._rows_for_cat([a, b, c])
    stacked = torch.cat(out, dim=0)
    assert stacked.shape == (3, 5)


def test_rows_for_cat_empty() -> None:
    assert ms._rows_for_cat([]) == []


def test_tensor_to_faiss_numpy_float32_contiguous() -> None:
    t = torch.randn(2, 10, dtype=torch.float64)
    arr = ms._tensor_to_faiss_numpy(t)
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]


def test_resolve_config_file_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="Index config not found"):
        ms._resolve_config_file(str(missing))


def test_abs_repo_path_relative_to_repo() -> None:
    p = ms._abs_repo_path("README.md")
    assert Path(p).is_file()
    assert Path(p).name == "README.md"
