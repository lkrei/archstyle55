
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import RESULTS_DIR, SPLITS_DIR
from .class_aliases import apply_aliases

Image.MAX_IMAGE_PIXELS = None


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_gallery(samples, logits, labels, class_names, n: int, out_path: Path,
                  thumb_size: int = 256) -> list[dict]:
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    errors_idx = np.where(preds != labels)[0]
    if len(errors_idx) == 0:
        return []

    errors_idx = errors_idx[np.argsort(-confidence[errors_idx])][:n]
    cols = 4
    rows = (len(errors_idx) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + 60)), (255, 255, 255))
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()

    records: list[dict] = []
    for i, idx in enumerate(errors_idx):
        sample = samples[int(idx)]
        true_name = class_names[int(labels[idx])]
        pred_name = class_names[int(preds[idx])]
        with Image.open(sample["path"]).convert("RGB") as img:
            img.thumbnail((thumb_size, thumb_size))
            row, col = divmod(i, cols)
            x = col * thumb_size + (thumb_size - img.size[0]) // 2
            y = row * (thumb_size + 60) + (thumb_size - img.size[1]) // 2
            canvas.paste(img, (x, y))
        draw = ImageDraw.Draw(canvas)
        ty = (i // cols) * (thumb_size + 60) + thumb_size + 4
        tx = (i % cols) * thumb_size + 4
        draw.text((tx, ty), f"true: {true_name}", fill=(0, 100, 0), font=font)
        draw.text((tx, ty + 18), f"pred: {pred_name} (p={confidence[idx]:.2f})",
                  fill=(150, 0, 0), font=font)

        records.append({
            "path": sample["path"],
            "true": true_name, "pred": pred_name,
            "confidence": float(confidence[idx]),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return records


def top_pairs(labels: np.ndarray, preds: np.ndarray, class_names: list[str], k: int) -> list[dict]:
    counter = Counter()
    for t, p in zip(labels, preds):
        if t != p:
            counter[(int(t), int(p))] += 1
    out = []
    for (t, p), n in counter.most_common(k):
        out.append({"true": class_names[t], "pred": class_names[p], "count": int(n)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--classes", type=Path, default=SPLITS_DIR / "idx_to_class.json",
                        help="idx_to_class.json (если в logits.npz нет class_names)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "errors")
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--top-pairs", type=int, default=20)
    args = parser.parse_args()

    bundle = np.load(args.logits, allow_pickle=True)
    logits = bundle["logits"]
    labels = bundle["labels"].astype(np.int64)
    if "class_names" in bundle.files:
        class_names_raw = list(bundle["class_names"])
    else:
        idx = json.loads(args.classes.read_text(encoding="utf-8"))
        class_names_raw = [idx[str(i)] for i in range(len(idx))]
    class_names = apply_aliases(class_names_raw)
    splits = json.loads(args.splits.read_text())
    samples = splits["test"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = build_gallery(samples, logits, labels, class_names, args.n_samples,
                            args.out_dir / "error_gallery.png")
    pairs = top_pairs(labels, logits.argmax(axis=1), class_names, args.top_pairs)
    (args.out_dir / "error_gallery.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "top_error_pairs.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"gallery: {len(records)} | top pairs: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
