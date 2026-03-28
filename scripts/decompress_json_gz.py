"""Decompress every *.json.gz in a directory to plain .json files in an output directory."""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path


def decompress_json_gz(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or not path.name.endswith(".json.gz"):
            continue
        out_path = output_dir / path.name[:-3]
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f_in:
                with out_path.open("wt", encoding="utf-8") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Распакован {path} -> {out_path}")
            count += 1
        except OSError as e:
            print(f"Ошибка {path}: {e}", file=sys.stderr)
    if count == 0:
        print(f"Не найдено .json.gz в {input_dir}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Каталог с файлами *.json.gz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Каталог для распакованных *.json",
    )
    args = p.parse_args()
    inp = args.input_dir.resolve()
    if not inp.is_dir():
        print(f"Не каталог: {inp}", file=sys.stderr)
        sys.exit(1)
    decompress_json_gz(inp, args.output_dir.resolve())


if __name__ == "__main__":
    main()
