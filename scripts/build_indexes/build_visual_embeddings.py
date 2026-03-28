"""Build ColQwen multi-vector embedding shards (.pt) for images under images_path."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import unicodedata
from pathlib import Path

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from dotenv import load_dotenv
from omegaconf import OmegaConf
from PIL import Image

_CHUNK = 500


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_embeddings(config_path: Path, device: str | None) -> None:
    root = _project_root()
    os.chdir(root)
    cfg = OmegaConf.load(config_path)
    required = ["model_name", "images_path", "embeddings_path"]
    if not all(k in cfg for k in required):
        print("Config missing required keys", file=sys.stderr)
        sys.exit(1)

    images_path = Path(cfg.images_path)
    emb_dir = Path(cfg.embeddings_path)
    if not images_path.is_dir():
        print(f"images_path is not a directory: {images_path}", file=sys.stderr)
        sys.exit(1)
    emb_dir.mkdir(parents=True, exist_ok=True)

    dev = device or _pick_device()
    model = ColQwen2.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map=dev,
    )
    processor = ColQwen2Processor.from_pretrained(cfg.model_name)

    pdf_names = sorted(
        d for d in os.listdir(images_path) if (images_path / d).is_dir()
    )
    all_tensors: list[torch.Tensor] = []

    for pdf_name in pdf_names:
        pdf_name_nfc = unicodedata.normalize("NFC", pdf_name)
        sub = images_path / pdf_name
        for image_path in sorted(sub.glob("*.png")):
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    batch = processor.process_images(img).to(model.device)
                    with torch.no_grad():
                        out = model(**batch).to(torch.bfloat16)
                    _clear_cache()
                    all_tensors.append(out.cpu())
            except OSError as e:
                print(f"Skip {image_path}: {e}", file=sys.stderr)

    if not all_tensors:
        print("No embeddings produced.", file=sys.stderr)
        sys.exit(1)

    stacked = torch.cat(all_tensors, dim=0)

    for old in glob.glob(str(emb_dir / "embeddings_*.pt")):
        os.remove(old)

    for i in range(0, stacked.shape[0], _CHUNK):
        chunk = stacked[i : i + _CHUNK]
        out_path = emb_dir / f"embeddings_{i:08d}.pt"
        torch.save(chunk, out_path)
        print(f"Wrote {out_path} shape={tuple(chunk.shape)}")

    print(f"Done: {stacked.shape[0]} embedding rows in {emb_dir}")


def main() -> None:
    load_dotenv(override=True)
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to visual_index.yaml (default: VISUAL_INDEX_CONFIG_PATH)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda | mps | cpu (default: auto)",
    )
    args = p.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        env = os.getenv("VISUAL_INDEX_CONFIG_PATH") or os.getenv("COLQWEN_CONFIG_PATH")
        if not env:
            print("Set VISUAL_INDEX_CONFIG_PATH or pass --config", file=sys.stderr)
            sys.exit(1)
        cfg_path = Path(env)
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    build_embeddings(cfg_path, args.device)


if __name__ == "__main__":
    main()
