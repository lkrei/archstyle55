
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="папка с подпапками-классами")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mapping-path", type=Path, default=None)
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 2

    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    skipped_dotfiles = 0
    conflicts: list[str] = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        files = []
        for f in class_dir.iterdir():
            if f.name.startswith("."):
                skipped_dotfiles += 1
                continue
            if f.is_file() and f.suffix.lower() in EXTS:
                files.append(f)
        files.sort(key=lambda p: p.name)

        prefix = slug(class_dir.name)
        for i, f in enumerate(files):
            ext = f.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            new_name = f"{prefix}_{i:05d}{ext}"
            new_path = f.with_name(new_name)
            if new_path == f:
                continue
            if new_path.exists():
                conflicts.append(f"{f} -> {new_path}")
                continue
            if not args.dry_run:
                f.rename(new_path)
            rel_new = f"{class_dir.name}/{new_name}"
            mapping[rel_new] = f.name

        counts[class_dir.name] = len(files)

    mapping_path = args.mapping_path or (root.parent / "rename_mapping.json")
    if not args.dry_run:
        mapping_path.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total = sum(counts.values())
    print(f"classes: {len(counts)}")
    print(f"total files: {total}")
    print(f"renamed: {len(mapping)}")
    print(f"skipped dotfiles: {skipped_dotfiles}")
    if conflicts:
        print(f"conflicts: {len(conflicts)}")
        for c in conflicts[:10]:
            print("  ", c)
    if args.dry_run:
        print("(dry run — nothing was changed)")
    else:
        print(f"mapping written to: {mapping_path}")
    return 0 if not conflicts else 1


if __name__ == "__main__":
    raise SystemExit(main())
