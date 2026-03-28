"""Submodule vs package exports (name collision on `mistral_api.chat`)."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_import_chat_submodule_via_importlib() -> None:
    m = importlib.import_module("src.mistral_api.chat")
    assert m.__name__ == "src.mistral_api.chat"
    assert callable(m.chat)
    assert hasattr(m, "Mistral")


@pytest.fixture()
def fresh_mistral_api_package():
    """Drop mistral_api* from sys.modules so `from … import chat` hits the function."""
    keys = [k for k in sys.modules if k == "src.mistral_api" or k.startswith("src.mistral_api.")]
    saved = {k: sys.modules[k] for k in keys}
    for k in keys:
        del sys.modules[k]
    yield
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules.update(saved)


def test_from_package_import_chat_is_callable(fresh_mistral_api_package) -> None:
    from src.mistral_api import chat

    assert callable(chat)
