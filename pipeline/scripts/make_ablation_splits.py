
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--drop", action="append", required=True,
                        help="имя класса для удаления (можно несколько раз)")
    args = parser.parse_args()

    drop_set = set(args.drop)
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    src_idx = json.loads((args.src_dir / "idx_to_class.json").read_text(encoding="utf-8"))
    src_splits = json.loads((args.src_dir / "data_splits.json").read_text(encoding="utf-8"))

    keep_old_idx = []
    keep_names = []
    for i_str in sorted(src_idx, key=int):
        name = src_idx[i_str]
        if name in drop_set:
            continue
        keep_old_idx.append(int(i_str))
        keep_names.append(name)
    if not keep_names:
        raise SystemExit("nothing to keep")

    old_to_new = {old: new for new, old in enumerate(keep_old_idx)}
    new_idx_to_class = {str(new): name for new, name in enumerate(keep_names)}
    new_class_to_idx = {name: new for new, name in enumerate(keep_names)}

    new_splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    drop_counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        for s in src_splits[split]:
            old = int(s["label"])
            if old not in old_to_new:
                drop_counts[split] += 1
                continue
            new_splits[split].append({"path": s["path"], "label": old_to_new[old]})

    (args.dst_dir / "data_splits.json").write_text(
        json.dumps(new_splits, ensure_ascii=False), encoding="utf-8",
    )
    (args.dst_dir / "idx_to_class.json").write_text(
        json.dumps(new_idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.dst_dir / "class_to_idx.json").write_text(
        json.dumps(new_class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    src_manifest = args.src_dir / "manifest.csv"
    if src_manifest.is_file():
        with src_manifest.open(newline="", encoding="utf-8") as fin, \
             (args.dst_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["class_name"] in drop_set:
                    continue
                row["label"] = str(old_to_new[int(row["label"])])
                writer.writerow(row)

    splits_hash = hashlib.sha256(
        (args.dst_dir / "data_splits.json").read_bytes()
    ).hexdigest()[:16]

    meta = {
        "src_dir": str(args.src_dir),
        "dropped_classes": sorted(drop_set),
        "old_num_classes": len(src_idx),
        "new_num_classes": len(keep_names),
        "kept_classes_first5": keep_names[:5],
        "kept_classes_last5": keep_names[-5:],
        "counts": {k: len(v) for k, v in new_splits.items()},
        "dropped_counts": drop_counts,
        "splits_sha256_16": splits_hash,
    }
    (args.dst_dir / "ablation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
