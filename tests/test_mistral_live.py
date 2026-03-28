"""Live call to Mistral API (requires .env)."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)


@pytest.mark.skipif(not os.getenv("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set")
@pytest.mark.skipif(not os.getenv("MODEL_NAME"), reason="MODEL_NAME not set")
@pytest.mark.skipif(not os.getenv("PROMPTS_PATH"), reason="PROMPTS_PATH not set")
def test_summarize_image_live(tmp_path: Path) -> None:
    from src.mistral_api import summarize_image

    img_path = tmp_path / "t.png"
    Image.new("RGB", (64, 48), color=(120, 80, 200)).save(img_path)
    text = summarize_image(str(img_path))
    assert isinstance(text, str)
    assert text.strip()
