"""Unit tests for src.mistral_api.chat with mocked HTTP client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import importlib
import pytest

# `import src.mistral_api.chat` binds the re-exported `chat` function on the package;
# importlib.import_module returns the actual chat.py module (needed to patch `Mistral`).
chat_mod = importlib.import_module("src.mistral_api.chat")


@pytest.fixture
def prompts_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.yaml"
    p.write_text(
        "system: You are a test bot.\nsummary: Summarize.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture(autouse=True)
def reset_chat_module_state() -> None:
    chat_mod._client = None
    chat_mod._prompts = None
    chat_mod._model = None
    yield
    chat_mod._client = None
    chat_mod._prompts = None
    chat_mod._model = None


def test_chat_appends_images_to_last_user_message(
    monkeypatch: pytest.MonkeyPatch,
    prompts_yaml: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("PROMPTS_PATH", str(prompts_yaml))

    img = tmp_path / "x.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    fake_msg = MagicMock()
    fake_msg.content = "Ответ модели"
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.complete.return_value = fake_resp

    monkeypatch.setattr(chat_mod, "Mistral", lambda api_key: fake_client)

    history = [{"role": "user", "content": [{"type": "text", "text": "Вопрос"}]}]
    out = chat_mod.chat(history, images=[str(img)])

    assert out == "Ответ модели"
    fake_client.chat.complete.assert_called_once()
    kwargs = fake_client.chat.complete.call_args.kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    user_blocks = messages[-1]["content"]
    types = [b.get("type") for b in user_blocks]
    assert "text" in types
    assert "image_url" in types


def test_chat_empty_choices_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
    prompts_yaml: Path,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("PROMPTS_PATH", str(prompts_yaml))

    fake_resp = MagicMock()
    fake_resp.choices = []
    fake_client = MagicMock()
    fake_client.chat.complete.return_value = fake_resp
    monkeypatch.setattr(chat_mod, "Mistral", lambda api_key: fake_client)

    assert chat_mod.chat([{"role": "user", "content": [{"type": "text", "text": "q"}]}]) == ""


def test_summarize_image_returns_str(
    monkeypatch: pytest.MonkeyPatch,
    prompts_yaml: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("PROMPTS_PATH", str(prompts_yaml))

    img = tmp_path / "p.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    fake_msg = MagicMock()
    fake_msg.content = "Краткое описание"
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=fake_msg)]

    fake_client = MagicMock()
    fake_client.chat.complete.return_value = fake_resp
    monkeypatch.setattr(chat_mod, "Mistral", lambda api_key: fake_client)

    assert chat_mod.summarize_image(str(img)) == "Краткое описание"
