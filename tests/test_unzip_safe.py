"""Safe unzip behavior for scripts/prepare_documents/zip_or_unzip_folder.py."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


def _load_zip_tool():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/prepare_documents/zip_or_unzip_folder.py"
    spec = importlib.util.spec_from_file_location("zip_tool", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unzip_rejects_path_traversal(tmp_path: Path) -> None:
    mod = _load_zip_tool()
    zpath = tmp_path / "evil.zip"
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../../outside.txt", b"pwnd")
    with pytest.raises(ValueError, match="escapes destination"):
        mod.unzip_to_folder(zpath, out)


def test_zip_and_unzip_roundtrip(tmp_path: Path) -> None:
    mod = _load_zip_tool()
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.txt").write_text("data", encoding="utf-8")
    zpath = tmp_path / "bundle.zip"
    mod.zip_folder(src, zpath)
    dest = tmp_path / "dest"
    mod.unzip_to_folder(zpath, dest)
    assert (dest / "hello.txt").read_text(encoding="utf-8") == "data"
