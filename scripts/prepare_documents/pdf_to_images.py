"""Render each PDF in a folder to PNGs under output_folder/<pdf_stem>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf2image import convert_from_path


def convert_pdfs_to_images(pdf_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files in {pdf_folder}", file=sys.stderr)
        return

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        pdf_output = output_folder / pdf_name
        pdf_output.mkdir(parents=True, exist_ok=True)
        images = convert_from_path(str(pdf_path))
        for i, image in enumerate(images):
            out = pdf_output / f"{pdf_name}_page_{i}.png"
            image.save(out, "PNG")
    print("Преобразование завершено.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf-dir", type=Path, required=True, help="Folder with .pdf files")
    p.add_argument("--output-dir", type=Path, required=True, help="Root folder for PNG trees")
    args = p.parse_args()
    pdf_dir = args.pdf_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not pdf_dir.is_dir():
        print(f"Not a directory: {pdf_dir}", file=sys.stderr)
        sys.exit(1)
    convert_pdfs_to_images(pdf_dir, out_dir)


if __name__ == "__main__":
    main()
