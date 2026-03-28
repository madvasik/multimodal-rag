import glob
import json
import os
import re
from pathlib import Path
from typing import List
import unicodedata

import faiss
import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from dotenv import load_dotenv
from omegaconf import OmegaConf
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer

from src.mistral_api import summarize_image
from .retriever_base import BaseRetriever

load_dotenv(override=True)

_text_cfg = os.getenv("TEXT_INDEX_CONFIG_PATH") or os.getenv("BGE_CONFIG_PATH")
_visual_cfg = os.getenv("VISUAL_INDEX_CONFIG_PATH") or os.getenv("COLQWEN_CONFIG_PATH")
if not _text_cfg or not _visual_cfg:
    raise RuntimeError(
        "Set TEXT_INDEX_CONFIG_PATH and VISUAL_INDEX_CONFIG_PATH "
        "(или устаревшие BGE_CONFIG_PATH / COLQWEN_CONFIG_PATH)."
    )
text_index = OmegaConf.load(_text_cfg)
visual_index = OmegaConf.load(_visual_cfg)


def _sorted_embedding_shard_paths(embeddings_dir: str) -> List[str]:
    files = glob.glob(os.path.join(embeddings_dir, "embeddings_*.pt"))

    def sort_key(path: str) -> int:
        m = re.search(r"embeddings_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(files, key=sort_key)


def _load_shard_rows(path: str) -> List[torch.Tensor]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Unexpected object in {path}: {type(data)}")
    if data.dim() == 0:
        return [data]
    return list(data.unbind(0))


def _rows_for_cat(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Ensure each tensor has leading batch dim so torch.cat(..., dim=0) is valid."""
    if not embeddings:
        return []
    out: List[torch.Tensor] = []
    for t in embeddings:
        if t.dim() >= 2 and t.shape[0] == 1:
            out.append(t)
        elif t.dim() >= 1:
            out.append(t.unsqueeze(0))
        else:
            out.append(t.view(1, *t.shape))
    return out


def _tensor_to_faiss_numpy(embedding: torch.Tensor) -> np.ndarray:
    """FAISS Python API expects float32 numpy, contiguous."""
    return np.ascontiguousarray(embedding.detach().cpu().numpy().astype(np.float32))


class BGERetriever(BaseRetriever):
    def __init__(self, device: str = "cpu"):
        self.device = device
        if not os.path.isfile(text_index.faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found: {text_index.faiss_path}. "
                "Run: python scripts/build_indexes/build_text_faiss_index.py"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(text_index.model_name)
        self.model = AutoModel.from_pretrained(text_index.model_name).to(self.device)
        self.faiss_index = faiss.read_index(text_index.faiss_path)
        with open(text_index.metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.faiss_index.ntotal != len(self.meta):
            raise RuntimeError(
                f"FAISS ntotal ({self.faiss_index.ntotal}) != len(metadata) ({len(self.meta)}). "
                "Rebuild the text index."
            )

    def embed_queries(self, query: str | List[str]) -> torch.tensor:
        if isinstance(query, str):
            query = [query]
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings.float().cpu()
        return embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embed_queries(query)
        q = _tensor_to_faiss_numpy(query_embedding)
        _, indices = self.faiss_index.search(q, k=top_k)
        rows = indices[0]
        paths: List[str] = []
        for i in rows:
            if i < 0 or i >= len(self.meta):
                continue
            meta = self.meta[i]
            paths.append(
                os.path.join(
                    text_index.images_path,
                    unicodedata.normalize("NFC", meta["pdf"]),
                    meta["jpeg"],
                )
            )
        return paths

    def _add_image_to_index(self, image_path: str) -> None:
        summary = summarize_image(image_path)
        embedding = self.embed_queries(summary)
        self.faiss_index.add(_tensor_to_faiss_numpy(embedding))
        faiss.write_index(self.faiss_index, text_index.faiss_path)
        pdf_name = image_path.split("/")[-2]
        pdf_name = unicodedata.normalize("NFC", pdf_name)
        image = image_path.split("/")[-1]
        self.meta.append({"pdf": pdf_name, "jpeg": image})
        with open(text_index.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)


class ColQwenRetriever:
    def __init__(self, device: str = "cpu"):
        self.device = device
        dtype = torch.float32
        self.model = ColQwen2.from_pretrained(
            visual_index.model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.chunk_size = 500
        self.top_k = visual_index.top_k
        self.processor = ColQwen2Processor.from_pretrained(visual_index.model_name)
        with open(visual_index.metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embeddings: List[torch.Tensor] = []
        emb_dir = visual_index.embeddings_path
        for fp in _sorted_embedding_shard_paths(emb_dir):
            self.embeddings.extend(_load_shard_rows(fp))

    def embed_queries(self, query: str | List[str]) -> torch.tensor:
        if isinstance(query, str):
            query = [query]
        batch_queries = self.processor.process_queries(query).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**batch_queries).to(torch.float32)
        return outputs.cpu()

    def embed_image(self, image: Image.Image) -> torch.tensor:
        batch_images = self.processor.process_images(image).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**batch_images).to(torch.float32)
        return outputs.cpu()

    def retrieve(self, query: str, top_k: int | None = None) -> List[str]:
        if not self.embeddings or not self.meta:
            return []
        k = self.top_k if top_k is None else top_k
        k = min(k, len(self.embeddings), len(self.meta))
        if k <= 0:
            return []
        query_embedding = self.embed_queries(query)
        scores = self.processor.score_multi_vector(query_embedding, self.embeddings)
        top_k_docs = scores.argsort(axis=1)[0][-k:]
        top_k_docs = torch.flip(top_k_docs, dims=[0]).tolist()
        metas = [self.meta[i] for i in top_k_docs]
        return [
            os.path.join(
                visual_index.images_path,
                unicodedata.normalize("NFC", meta["pdf"]),
                meta["jpeg"],
            )
            for meta in metas
        ]

    def _add_image_to_index(self, image_path: str) -> None:
        try:
            with open(image_path, "rb") as f:
                img = Image.open(f)
                embedding = self.embed_image(img)
            self.embeddings.append(embedding)
            self._save_embeddings()
            pdf_name = image_path.split("/")[-2]
            pdf_name = unicodedata.normalize("NFC", pdf_name)
            jpeg_name = image_path.split("/")[-1]
            self.meta.append({"pdf": pdf_name, "jpeg": jpeg_name})
            with open(visual_index.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)
        except FileNotFoundError:
            pass

    def _save_embeddings(self) -> None:
        emb_dir = visual_index.embeddings_path
        os.makedirs(emb_dir, exist_ok=True)
        if not self.embeddings:
            return
        stacked = torch.cat(_rows_for_cat(self.embeddings), dim=0)
        for old in glob.glob(os.path.join(emb_dir, "embeddings_*.pt")):
            os.remove(old)
        for start in range(0, stacked.shape[0], self.chunk_size):
            chunk = stacked[start : start + self.chunk_size]
            out_path = os.path.join(emb_dir, f"embeddings_{start:08d}.pt")
            torch.save(chunk, out_path)


class RetrievePipeline:
    def __init__(self, device: str = "cpu"):
        self.text_retriever = BGERetriever(device=device)
        self.visual_retriever = ColQwenRetriever(device=device)

    def retrieve(self, query: str, strategy: str = "ColQwen+SummaryEmb"):
        if strategy == "SummaryEmb":
            images = self.text_retriever.retrieve(query)
        elif strategy == "ColQwen":
            images = self.visual_retriever.retrieve(query)
        elif strategy == "ColQwen+SummaryEmb":
            visual_hits = self.visual_retriever.retrieve(query, top_k=3)
            text_hits = self.text_retriever.retrieve(query, top_k=3)
            combined_images = visual_hits + text_hits
            images = list(dict.fromkeys(combined_images))
        else:
            images = []
        return images

    def add_to_index(self, pdf_path: str) -> None:
        p = Path(pdf_path).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() != ".pdf":
            raise ValueError(f"Expected path to a .pdf file, got: {pdf_path}")

        stem = unicodedata.normalize("NFC", p.stem)
        img_root = Path(text_index.images_path).expanduser().resolve() / stem
        img_root.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(p))
        for i, page in enumerate(pages):
            out = img_root / f"{stem}_page_{i}.png"
            page.save(out, "PNG")
            path_str = str(out)
            self.text_retriever._add_image_to_index(path_str)
            self.visual_retriever._add_image_to_index(path_str)
