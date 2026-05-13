
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD
from ..models.factory import DEFAULT_HPARAMS, TVBackboneClassifier, _replace_classifier
from ..xai.cam import grad_cam_pp
from .class_aliases import apply_aliases


def _build_arch_no_weights(name: str, num_classes: int, dropout: float = 0.1) -> torch.nn.Module:
    """Строим архитектуру TVBackboneClassifier без pretrained-весов."""
    from torchvision import models as tvm

    if name == "resnet50":
        m = tvm.resnet50(weights=None);          attr = "fc"
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None);   attr = "classifier"
    elif name == "efficientnet_b2":
        m = tvm.efficientnet_b2(weights=None);   attr = "classifier"
    elif name == "efficientnet_b3":
        m = tvm.efficientnet_b3(weights=None);   attr = "classifier"
    elif name == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=None); attr = "classifier"
    elif name == "convnext_small":
        m = tvm.convnext_small(weights=None);    attr = "classifier"
    elif name == "vit_b16":
        m = tvm.vit_b_16(weights=None);          attr = "heads"
    elif name == "swin_v2_t":
        m = tvm.swin_v2_t(weights=None);         attr = "head"
    else:
        raise ValueError(f"unsupported model: {name}")
    m, in_f = _replace_classifier(m, attr, num_classes, dropout)
    return TVBackboneClassifier(m, in_f, num_classes, dropout)

Image.MAX_IMAGE_PIXELS = None


def _last_conv(model: torch.nn.Module) -> torch.nn.Module:
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, torch.nn.Conv2d):
            return mod
    raise RuntimeError("no Conv2d found")


def _overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = np.clip(cam, 0, 1)
    cmap = plt.get_cmap("jet")
    heat = (cmap(cam)[..., :3] * 255.0).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-wrong", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                             "cpu")
    print(f"  device: {device}")

    spec = DEFAULT_HPARAMS[args.model]
    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    num_classes = len(idx_to_class)
    class_names = apply_aliases([idx_to_class[str(i)] for i in range(num_classes)])

    model = _build_arch_no_weights(args.model, num_classes=num_classes)
    ckpt = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state", "model_state", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise SystemExit(f"checkpoint mismatch: missing={len(missing)}, "
                         f"unexpected={len(unexpected)}; first missing: {missing[:3]}")
    if missing:
        print(f"  note: {len(missing)} missing keys: {missing}")
    model = model.to(device).eval()

    npz = args.run_dir / "test_logits.npz"
    data = np.load(npz, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    preds = logits.argmax(axis=1)
    confidence = (np.exp(logits - logits.max(axis=1, keepdims=True))
                  / np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
                  ).max(axis=1)

    splits = json.loads(args.splits.read_text())
    test_samples = splits["test"]
    if len(test_samples) != len(labels):
        raise SystemExit(f"size mismatch: splits={len(test_samples)} vs logits={len(labels)}")

    rng = np.random.default_rng(args.seed)
    correct_idx = np.where(preds == labels)[0]
    wrong_idx = np.where(preds != labels)[0]
    correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][:args.n_correct]
    wrong_idx = wrong_idx[np.argsort(-confidence[wrong_idx])][:args.n_wrong]

    tf = transforms.Compose([
        transforms.Resize(spec.image_size + 32),
        transforms.CenterCrop(spec.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    target_layer = _last_conv(model)

    rows = max(args.n_correct, args.n_wrong)
    fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 3.4))
    if rows == 1:
        axes = axes[None, :]

    def fill_column(idxs, col_offset, label_kind):
        for r, i in enumerate(idxs):
            sample = test_samples[int(i)]
            with Image.open(sample["path"]).convert("RGB") as im:
                rgb_disp = np.asarray(im.resize((spec.image_size, spec.image_size)))
            x = tf(Image.open(sample["path"]).convert("RGB")).to(device)
            cam = grad_cam_pp(model, x, target_layer, target_class=int(preds[i]))
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (spec.image_size, spec.image_size),
                    resample=Image.BILINEAR
                )
            ) / 255.0
            ov = _overlay(rgb_disp, cam_resized)

            true_n = class_names[int(labels[i])]
            pred_n = class_names[int(preds[i])]
            title_top = (f"true: {true_n}\n"
                         f"pred: {pred_n}  p={confidence[i]:.2f}")

            ax_img = axes[r, col_offset]
            ax_ov  = axes[r, col_offset + 1]
            ax_img.imshow(rgb_disp); ax_img.set_xticks([]); ax_img.set_yticks([])
            ax_ov.imshow(ov);        ax_ov.set_xticks([]);  ax_ov.set_yticks([])
            ax_img.set_title(title_top, fontsize=8,
                             color=("#0a3" if label_kind == "ok" else "#a30"))

    fill_column(correct_idx, 0, "ok")
    fill_column(wrong_idx, 2, "bad")

    for r in range(rows):
        if r == 0:
            axes[r, 0].set_ylabel("CORRECT", rotation=90, fontsize=10, labelpad=10)
            axes[r, 2].set_ylabel("WRONG",   rotation=90, fontsize=10, labelpad=10)

    fig.suptitle(f"Grad-CAM++ on {args.model} — top-confidence correct vs wrong",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"xai_showcase_{args.model}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
