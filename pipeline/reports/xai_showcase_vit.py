
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD
from ..models.factory import DEFAULT_HPARAMS, TVBackboneClassifier, _replace_classifier
from .class_aliases import apply_aliases

Image.MAX_IMAGE_PIXELS = None


def _build_arch_no_weights(name: str, num_classes: int, dropout: float = 0.1) -> torch.nn.Module:
    from torchvision import models as tvm
    if name == "vit_b16":
        m = tvm.vit_b_16(weights=None);  attr = "heads"
    elif name == "swin_v2_t":
        m = tvm.swin_v2_t(weights=None); attr = "head"
    else:
        raise ValueError(f"unsupported model: {name}")
    m, in_f = _replace_classifier(m, attr, num_classes, dropout)
    return TVBackboneClassifier(m, in_f, num_classes, dropout)


@contextmanager
def _patch_vit_for_attn(model: torch.nn.Module, *, keep_grad: bool = False):
    """Временно меняет forward у EncoderBlock'ов torchvision-ViT,
    чтобы self-attention возвращал attn_weights. Каждый блок кладёт
    attn в общий cache и возвращает только тензор. Если ``keep_grad``,
    то attn кладётся БЕЗ ``.detach()`` (для Chefer-relevance).
    """
    from torchvision.models.vision_transformer import EncoderBlock

    blocks = [m for m in model.modules() if isinstance(m, EncoderBlock)]
    saved = [b.forward for b in blocks]
    cache: list[torch.Tensor] = []

    def _make_forward(orig_self):
        def fwd(input):
            x = orig_self.ln_1(input)
            x, attn = orig_self.self_attention(x, x, x, need_weights=True,
                                               average_attn_weights=False)
            cache.append(attn if keep_grad else attn.detach())
            x = orig_self.dropout(x)
            x = x + input
            y = orig_self.ln_2(x)
            y = orig_self.mlp(y)
            return x + y
        return fwd

    for b in blocks:
        b.forward = _make_forward(b)
    try:
        yield cache
    finally:
        for b, f in zip(blocks, saved):
            b.forward = f


def _rollout_from_cache(cache: list[torch.Tensor]) -> np.ndarray:
    """Реализация attention rollout прямо из готового кэша.

    Каждый attn — (B, heads, T, T). Усредняем по головам, добавляем
    единичную, нормализуем по строкам, перемножаем.
    """
    result = None
    for a in cache:
        a = a.mean(dim=1)
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else torch.matmul(a, result)
    cls = result[0, 0, 1:].cpu().numpy()
    side = int(np.sqrt(cls.shape[0]))
    return cls.reshape(side, side)


def _chefer_from_grads(attns: list[torch.Tensor],
                       grads: list[torch.Tensor]) -> np.ndarray:
    result = None
    for attn, grad in zip(attns, grads):
        weighted = (attn * grad).clamp(min=0).mean(dim=1)
        weighted = weighted + torch.eye(weighted.size(-1),
                                        device=weighted.device).unsqueeze(0)
        weighted = weighted / weighted.sum(dim=-1, keepdim=True)
        result = weighted if result is None else torch.matmul(weighted, result)
    cls = result[0, 0, 1:].detach().cpu().numpy()
    side = int(np.sqrt(cls.shape[0]))
    return cls.reshape(side, side)


def _overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    heat = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR)) / 255.0
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat)[..., :3] * 255).astype(np.uint8)
    return (alpha * heat_rgb + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="vit_b16")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-wrong", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or "cpu"
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
                state = ckpt[key]; break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5:
        raise SystemExit(f"checkpoint mismatch: {len(missing)} missing")
    model = model.to(device).eval()

    npz = args.run_dir / "test_logits.npz"
    data = np.load(npz, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    preds  = logits.argmax(axis=1)
    confidence = (np.exp(logits - logits.max(axis=1, keepdims=True))
                  / np.exp(logits - logits.max(axis=1, keepdims=True))
                  .sum(axis=1, keepdims=True)).max(axis=1)

    splits = json.loads(args.splits.read_text())
    test_samples = splits["test"]
    if len(test_samples) != len(labels):
        raise SystemExit("size mismatch")

    correct_idx = np.where(preds == labels)[0]
    wrong_idx   = np.where(preds != labels)[0]
    correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][:args.n_correct]
    wrong_idx   = wrong_idx[np.argsort(-confidence[wrong_idx])][:args.n_wrong]

    tf = transforms.Compose([
        transforms.Resize(spec.image_size + 32),
        transforms.CenterCrop(spec.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    rows = max(args.n_correct, args.n_wrong)
    fig, axes = plt.subplots(rows, 6, figsize=(18, rows * 3.0))
    if rows == 1:
        axes = axes[None, :]

    def render(idxs, base_col, kind):
        for r, i in enumerate(idxs):
            sample = test_samples[int(i)]
            with Image.open(sample["path"]).convert("RGB") as im:
                rgb_disp = np.asarray(im.resize((spec.image_size, spec.image_size)))
            x = tf(Image.open(sample["path"]).convert("RGB")).to(device)

            with _patch_vit_for_attn(model, keep_grad=False) as cache:
                with torch.no_grad():
                    _ = model(x.unsqueeze(0))
                rollout = _rollout_from_cache(cache)

            with _patch_vit_for_attn(model, keep_grad=True) as cache:
                xx = x.unsqueeze(0).clone().detach().requires_grad_(True)
                logits_t = model(xx)
                target = int(preds[i])
                score = logits_t[0, target]
                grads = torch.autograd.grad(score, cache, retain_graph=False,
                                            allow_unused=True)
                grads_filled = [g if g is not None else torch.zeros_like(a)
                                for a, g in zip(cache, grads)]
                attns_det = [a.detach() for a in cache]
                chefer = _chefer_from_grads(attns_det, grads_filled)

            rollout = (rollout - rollout.min()) / max(rollout.max() - rollout.min(), 1e-9)
            chefer  = (chefer  - chefer.min())  / max(chefer.max()  - chefer.min(),  1e-9)
            ov_r = _overlay(rgb_disp, rollout)
            ov_c = _overlay(rgb_disp, chefer)

            true_n = class_names[int(labels[i])]
            pred_n = class_names[int(preds[i])]
            title  = f"true: {true_n}\npred: {pred_n}  p={confidence[i]:.2f}"

            ax_img = axes[r, base_col]
            ax_ro  = axes[r, base_col + 1]
            ax_ch  = axes[r, base_col + 2]
            ax_img.imshow(rgb_disp); ax_img.set_xticks([]); ax_img.set_yticks([])
            ax_ro.imshow(ov_r);      ax_ro.set_xticks([]);  ax_ro.set_yticks([])
            ax_ch.imshow(ov_c);      ax_ch.set_xticks([]);  ax_ch.set_yticks([])
            color = "#0a3" if kind == "ok" else "#a30"
            ax_img.set_title(title, fontsize=8, color=color)
            if r == 0:
                ax_img.set_xlabel("input", fontsize=9)
                ax_ro.set_xlabel("attention rollout", fontsize=9)
                ax_ch.set_xlabel("Chefer relevance", fontsize=9)

    render(correct_idx, 0, "ok")
    render(wrong_idx,   3, "bad")

    fig.suptitle(f"ViT XAI on {args.model} — top-confidence correct (left) vs wrong (right)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"xai_showcase_{args.model}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
