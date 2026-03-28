"""Build FAISS text index (BGE on Mistral page summaries). Run from repo root."""

import glob
import json
import os
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.mistral_api import summarize_image
except ImportError as e:
    print(f"Import error (run from repo root, pip install -e .): {e}", file=sys.stderr)
    sys.exit(1)

load_dotenv(override=True)

TEXT_CFG = os.getenv("TEXT_INDEX_CONFIG_PATH") or os.getenv("BGE_CONFIG_PATH")
if not TEXT_CFG:
    TEXT_CFG = str(_REPO_ROOT / "src/config/text_index.yaml")
if not os.path.isfile(TEXT_CFG):
    print(f"Text index config not found: {TEXT_CFG}", file=sys.stderr)
    sys.exit(1)

DEVICE = "cpu"


def build_index() -> None:
    try:
        text_index = OmegaConf.load(TEXT_CFG)
    except Exception:
        sys.exit(1)

    required_keys = ["model_name", "images_path", "faiss_path", "metadata_path"]
    if not all(key in text_index for key in required_keys):
        sys.exit(1)

    base_images_path = text_index.images_path
    if not os.path.isdir(base_images_path):
        sys.exit(1)

    try:
        pdf_names = [
            d
            for d in os.listdir(base_images_path)
            if os.path.isdir(os.path.join(base_images_path, d))
        ]
    except OSError:
        sys.exit(1)

    faiss_dir = os.path.dirname(text_index.faiss_path)
    meta_dir = os.path.dirname(text_index.metadata_path)
    try:
        os.makedirs(faiss_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
    except OSError:
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(text_index.model_name)
        model = AutoModel.from_pretrained(text_index.model_name).to(DEVICE)
        model.eval()
    except Exception:
        sys.exit(1)

    embedding_dim = model.config.hidden_size
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    all_embeddings = []
    total_images_processed = 0
    total_images_skipped = 0

    for pdf_name in pdf_names:
        source_image_dir = os.path.join(base_images_path, pdf_name)
        image_files = sorted(glob.glob(os.path.join(source_image_dir, "*.png")))
        num_images = len(image_files)

        if num_images == 0:
            continue

        current_doc_images_processed = 0

        for image_path in image_files:
            base_name = os.path.basename(image_path)
            try:
                summary = summarize_image(image_path)
                if not summary:
                    total_images_skipped += 1
                    continue

                inputs = tokenizer(
                    [summary],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)

                embedding = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embedding.float().cpu().numpy())

                jpeg_name = base_name
                pdf_name_normalized = unicodedata.normalize("NFC", pdf_name)
                metadata.append({"pdf": pdf_name_normalized, "jpeg": jpeg_name})
                current_doc_images_processed += 1

            except Exception:
                total_images_skipped += 1

        total_images_processed += current_doc_images_processed

    if all_embeddings:
        embeddings_matrix = np.vstack(all_embeddings)
        index.add(embeddings_matrix)
        try:
            faiss.write_index(index, text_index.faiss_path)
        except Exception:
            pass

        if len(metadata) == index.ntotal:
            try:
                with open(text_index.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
            except Exception:
                pass


if __name__ == "__main__":
    build_index()
