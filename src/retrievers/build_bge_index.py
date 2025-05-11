import os
import glob
import json
import faiss
import torch
import unicodedata
from transformers import AutoModel, AutoTokenizer
from omegaconf import OmegaConf
from dotenv import load_dotenv
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

try:
    from src.llm import summarize_image
except ImportError:
    sys.exit(1)

load_dotenv(override=True)

BGE_CONFIG_PATH = os.getenv("BGE_CONFIG_PATH")
if not BGE_CONFIG_PATH or not os.path.exists(BGE_CONFIG_PATH):
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_index():
    try:
        bge_config = OmegaConf.load(BGE_CONFIG_PATH)
    except Exception:
        sys.exit(1)

    required_keys = ["model_name", "images_path", "faiss_path", "metadata_path"]
    if not all(key in bge_config for key in required_keys):
        sys.exit(1)

    base_images_path = bge_config.images_path
    if not os.path.isdir(base_images_path):
        sys.exit(1)

    try:
        pdf_names = [d for d in os.listdir(base_images_path) if os.path.isdir(os.path.join(base_images_path, d))]
    except OSError:
        sys.exit(1)

    faiss_dir = os.path.dirname(bge_config.faiss_path)
    meta_dir = os.path.dirname(bge_config.metadata_path)
    try:
        os.makedirs(faiss_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
    except OSError:
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(bge_config.model_name)
        model = AutoModel.from_pretrained(bge_config.model_name).to(DEVICE)
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
                    [summary], return_tensors="pt", padding=True, truncation=True, max_length=512
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
            faiss.write_index(index, bge_config.faiss_path)
        except Exception:
            pass

        if len(metadata) == index.ntotal:
            try:
                with open(bge_config.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
            except Exception:
                pass
    else:
        pass

if __name__ == "__main__":
    build_index()