"""RetrievePipeline.retrieve strategies without loading BGE/ColQwen weights."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.retrieval.multimodal_search import RetrievePipeline


def _pipeline_with_mocks() -> RetrievePipeline:
    pipe = object.__new__(RetrievePipeline)
    pipe.text_retriever = MagicMock()
    pipe.visual_retriever = MagicMock()
    return pipe


def test_strategy_summary_emb_uses_text_only() -> None:
    pipe = _pipeline_with_mocks()
    pipe.text_retriever.retrieve.return_value = ["/a.png"]
    pipe.visual_retriever.retrieve.return_value = ["/b.png"]

    out = pipe.retrieve("q", "SummaryEmb")
    assert out == ["/a.png"]
    pipe.text_retriever.retrieve.assert_called_once_with("q")
    pipe.visual_retriever.retrieve.assert_not_called()


def test_strategy_colqwen_uses_visual_only() -> None:
    pipe = _pipeline_with_mocks()
    pipe.visual_retriever.retrieve.return_value = ["/v.png"]

    out = pipe.retrieve("q", "ColQwen")
    assert out == ["/v.png"]
    pipe.visual_retriever.retrieve.assert_called_once_with("q")
    pipe.text_retriever.retrieve.assert_not_called()


def test_strategy_hybrid_dedupes_order() -> None:
    pipe = _pipeline_with_mocks()
    pipe.visual_retriever.retrieve.return_value = ["/1.png", "/2.png"]
    pipe.text_retriever.retrieve.return_value = ["/2.png", "/3.png"]

    out = pipe.retrieve("q", "ColQwen+SummaryEmb")
    assert out == ["/1.png", "/2.png", "/3.png"]
    pipe.visual_retriever.retrieve.assert_called_once_with("q", top_k=3)
    pipe.text_retriever.retrieve.assert_called_once_with("q", top_k=3)


def test_strategy_unknown_returns_empty() -> None:
    pipe = _pipeline_with_mocks()
    assert pipe.retrieve("q", "NoSuchStrategy") == []
