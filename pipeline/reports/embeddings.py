
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD, RESULTS_DIR, SPLITS_DIR
from .class_aliases import apply_aliases

Image.MAX_IMAGE_PIXELS = None


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dinov2(device: str, model_id: str = "facebook/dinov2-base"):

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "need transformers: pip install transformers"
        ) from exc
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def encode(batch: torch.Tensor) -> torch.Tensor:
        out = model(pixel_values=batch)
        last = out.last_hidden_state
        return last[:, 0, :]

    return encode


@torch.no_grad()
def compute_embeddings(samples, image_size: int = 518, batch_size: int = 16,
                       device: str | None = None,
                       model_id: str = "facebook/dinov2-base") -> tuple[np.ndarray, np.ndarray]:
    device = _pick_device(device)
    encode = _load_dinov2(device, model_id=model_id)
    tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=f"DINOv2 features ({device})")
    except ImportError:
        iterator = samples

    feats, labels = [], []
    buffer = []
    buf_labels = []

    def flush():
        if not buffer:
            return
        batch = torch.stack(buffer).to(device)
        out = encode(batch).cpu().numpy()
        feats.append(out)
        labels.append(np.asarray(buf_labels, dtype=np.int64))
        buffer.clear(); buf_labels.clear()

    for s in iterator:
        try:
            img = Image.open(s["path"]).convert("RGB")
        except (OSError, Image.DecompressionBombError):
            continue
        buffer.append(tf(img))
        buf_labels.append(int(s["label"]))
        if len(buffer) >= batch_size:
            flush()
    flush()

    if not feats:
        raise RuntimeError("no embeddings computed")
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def class_centroids(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    dim = features.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids / np.maximum(norms, 1e-9)


def cosine_matrix(centroids: np.ndarray) -> np.ndarray:
    return centroids @ centroids.T


def umap_projection(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                    seed: int = 42) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        return reducer.fit_transform(features)
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(features)


def save_plots(
    out_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    cos_matrix: np.ndarray,
    proj: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroid_cosine.npy", cos_matrix)
    np.save(out_dir / "projection.npy", proj)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cos_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(out_dir / "centroid_cosine.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 11))
    palette = np.concatenate([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    for c in range(len(class_names)):
        m = labels == c
        if not m.any():
            continue
        ax.scatter(proj[m, 0], proj[m, 1], s=8, color=palette[c % len(palette)],
                   alpha=0.55, edgecolors="none")
    centroids2d = np.zeros((len(class_names), 2))
    for c in range(len(class_names)):
        m = labels == c
        if m.any():
            centroids2d[c] = proj[m].mean(axis=0)
    for c in range(len(class_names)):
        ax.text(centroids2d[c, 0], centroids2d[c, 1], str(c),
                fontsize=8, weight="bold", color="black",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.7,
                          edgecolor="none"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DINOv2 (frozen) embeddings on test set — UMAP")
    fig.tight_layout()
    fig.savefig(out_dir / "embedding_projection.png", dpi=200)
    plt.close(fig)

    legend_path = out_dir / "embedding_class_legend.txt"
    with legend_path.open("w", encoding="utf-8") as f:
        for c, name in enumerate(class_names):
            f.write(f"{c:3d}\t{name}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--classes", type=Path, default=SPLITS_DIR / "idx_to_class.json")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "embeddings")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-plots", action="store_true",
                        help="не строить cosine/UMAP plots (для train/val)")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    idx_to_class = json.loads(args.classes.read_text())
    num_classes = len(idx_to_class)
    class_names_raw = [idx_to_class[str(i)] for i in range(num_classes)]
    class_names = apply_aliases(class_names_raw)

    features, labels = compute_embeddings(
        splits[args.split], image_size=args.image_size, batch_size=args.batch_size,
        device=args.device, model_id=args.model_id,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = np.array([s["path"] for s in splits[args.split]
                      if "path" in s][:len(features)], dtype=object)
    np.savez_compressed(args.out_dir / f"embeddings_{args.split}.npz",
                        features=features, labels=labels,
                        paths=paths, class_names=np.array(class_names))
    print(f"saved features to {args.out_dir}/embeddings_{args.split}.npz "
          f"({features.shape})")

    if args.no_plots or args.split != "test":
        return 0

    centroids = class_centroids(features, labels, num_classes)
    cos = cosine_matrix(centroids)
    proj = umap_projection(features)
    save_plots(args.out_dir, features, labels, class_names, cos, proj)
    print(f"plots saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
