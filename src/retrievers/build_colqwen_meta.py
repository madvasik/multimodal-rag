import os
import glob
import json
import unicodedata
from omegaconf import OmegaConf
from dotenv import load_dotenv
import sys

load_dotenv(override=True)

COLQWEN_CONFIG_PATH = os.getenv("COLQWEN_CONFIG_PATH")
if not COLQWEN_CONFIG_PATH or not os.path.exists(COLQWEN_CONFIG_PATH):
    sys.exit(1)

def build_metadata():
    try:
        colqwen_config = OmegaConf.load(COLQWEN_CONFIG_PATH)
    except Exception:
        sys.exit(1)

    required_keys = ["images_path", "metadata_path", "embeddings_path"]
    if not all(key in colqwen_config for key in required_keys):
        sys.exit(1)

    base_images_path = colqwen_config.images_path
    if not os.path.isdir(base_images_path):
        sys.exit(1)

    try:
        pdf_names = [d for d in os.listdir(base_images_path) if os.path.isdir(os.path.join(base_images_path, d))]
    except OSError:
        sys.exit(1)

    meta_dir = os.path.dirname(colqwen_config.metadata_path)
    try:
        os.makedirs(meta_dir, exist_ok=True)
    except OSError:
        sys.exit(1)

    metadata = []
    total_images_processed = 0

    for pdf_name in pdf_names:
        source_image_dir = os.path.join(base_images_path, pdf_name)
        image_files = sorted(glob.glob(os.path.join(source_image_dir, "*.png")))
        num_images = len(image_files)

        if num_images == 0:
            continue

        total_images_processed += num_images

        for image_path in image_files:
            base_name = os.path.basename(image_path)
            jpeg_name = base_name
            pdf_name_normalized = unicodedata.normalize("NFC", pdf_name)
            metadata.append({"pdf": pdf_name_normalized, "jpeg": jpeg_name})

    if metadata:
        try:
            with open(colqwen_config.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    try:
        embedding_files = sorted(glob.glob(os.path.join(colqwen_config.embeddings_path, "*.pt")))
        if embedding_files:
            if len(metadata) != total_images_processed:
                print(f"Предупреждение: {len(metadata)} записей в метаданных, {total_images_processed} PNG файлов.")
        else:
            print("Предупреждение: Файлы эмбеддингов не найдены.")
    except Exception:
        pass

if __name__ == "__main__":
    build_metadata()
