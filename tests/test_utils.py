"""Tests for src.utils (no models, no API)."""

from pathlib import Path

import pytest
import yaml

from src.utils import (
    encode_image,
    image_data_url,
    image_mime_type,
    load_prompts_from_yaml,
)


def test_image_mime_type_png_jpeg_default() -> None:
    assert image_mime_type("x.PNG") == "image/png"
    assert image_mime_type("a.jpg") == "image/jpeg"
    assert image_mime_type("a.jpeg") == "image/jpeg"
    assert image_mime_type("noext") == "image/jpeg"


def test_image_data_url_uses_mime() -> None:
    url = image_data_url("f.png", "QUJD")
    assert url == "data:image/png;base64,QUJD"


def test_encode_image_roundtrip(tmp_path: Path) -> None:
    raw = b"\x00\x01\x02\xff"
    p = tmp_path / "bin.dat"
    p.write_bytes(raw)
    b64 = encode_image(p)
    assert b64 is not None
    import base64

    assert base64.b64decode(b64) == raw


def test_encode_image_path_str(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("hi")
    assert encode_image(str(p)) is not None


def test_encode_image_missing_returns_none() -> None:
    assert encode_image("/nonexistent/path/photo.png") is None


def test_load_prompts_from_yaml_ok(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text("system: hello\nsummary: world\n", encoding="utf-8")
    data = load_prompts_from_yaml(str(f))
    assert data == {"system": "hello", "summary": "world"}


def test_load_prompts_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_prompts_from_yaml(str(f))


def test_load_prompts_safe_loader_no_arbitrary_objects(tmp_path: Path) -> None:
    f = tmp_path / "unsafe.yaml"
    f.write_text(
        "!!python/object/apply:os.system ['echo unsafe']\n",
        encoding="utf-8",
    )
    with pytest.raises(yaml.constructor.ConstructorError):
        load_prompts_from_yaml(str(f))
