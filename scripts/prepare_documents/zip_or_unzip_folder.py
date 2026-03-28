"""Zip or unzip a directory tree (ZIP_DEFLATED)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def zip_folder(folder: Path, zip_path: Path) -> None:
    folder = folder.resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                arc = file_path.relative_to(folder)
                zf.write(file_path, arc)


def unzip_to_folder(zip_path: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(folder)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    z = sub.add_parser("zip", help="Pack a folder into a .zip file")
    z.add_argument("folder", type=Path, help="Directory to pack")
    z.add_argument("zipfile", type=Path, help="Output .zip path")

    u = sub.add_parser("unzip", help="Extract a .zip into a folder")
    u.add_argument("zipfile", type=Path, help="Input .zip path")
    u.add_argument("folder", type=Path, help="Destination directory")

    args = p.parse_args()
    if args.cmd == "zip":
        f = args.folder.resolve()
        if not f.is_dir():
            print(f"Not a directory: {f}", file=sys.stderr)
            sys.exit(1)
        args.zipfile.parent.mkdir(parents=True, exist_ok=True)
        zip_folder(f, args.zipfile.resolve())
        print(f"Packed {f} -> {args.zipfile.resolve()}")
    else:
        zpath = args.zipfile.resolve()
        if not zpath.is_file():
            print(f"Not a file: {zpath}", file=sys.stderr)
            sys.exit(1)
        unzip_to_folder(zpath, args.folder.resolve())
        print(f"Extracted {zpath} -> {args.folder.resolve()}")


if __name__ == "__main__":
    main()
