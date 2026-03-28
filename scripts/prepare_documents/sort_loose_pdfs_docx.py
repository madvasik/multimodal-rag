"""Разложить PDF и DOCX из корня каталога в подпапки pdf_files/ и docx_files/."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def organize(work_dir: Path, reset: bool) -> None:
    pdf_dir = work_dir / "pdf_files"
    docx_dir = work_dir / "docx_files"

    if reset:
        for d in (pdf_dir, docx_dir):
            if d.is_dir():
                shutil.rmtree(d)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)

    for path in list(work_dir.iterdir()):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf == ".pdf":
            path.rename(pdf_dir / path.name)
        elif suf == ".docx":
            path.rename(docx_dir / path.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing loose PDF/DOCX files (default: cwd)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing pdf_files/ and docx_files/ before organizing",
    )
    args = p.parse_args()
    work_dir = args.work_dir.resolve()
    if not work_dir.is_dir():
        print(f"Not a directory: {work_dir}", file=sys.stderr)
        sys.exit(1)
    organize(work_dir, args.reset)


if __name__ == "__main__":
    main()
