
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .class_aliases import apply_aliases

Image.MAX_IMAGE_PIXELS = None

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)


def _load_head(run_dir: Path, num_classes: int) -> torch.nn.Linear:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state", "model_state", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
    weight = state.get("head.fc.weight")
    bias   = state.get("head.fc.bias")
    if weight is None or bias is None:
        raise SystemExit("checkpoint must contain head.fc.weight/bias")
    head = torch.nn.Linear(weight.shape[1], num_classes)
    head.weight.data = weight.float()
    head.bias.data   = bias.float()
    return head.eval()


def _heat_to_overlay(rgb: np.ndarray, heat_grid: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    heat = (heat_grid - heat_grid.min()) / max(heat_grid.max() - heat_grid.min(), 1e-9)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR)
    heat_arr = np.asarray(heat_img, dtype=np.float64) / 255.0
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat_arr)[..., :3] * 255).astype(np.uint8)
    return (alpha * heat_rgb + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)


def _attentions_to_rollout(attns: list[torch.Tensor]) -> np.ndarray:
    """attns: list of (1, heads, T, T). Возвращает CLS→patches reshape (S, S)."""
    result = None
    for a in attns:
        a = a.mean(dim=1)
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else torch.matmul(a, result)
    cls_attn = result[0, 0, 1:].detach().cpu().numpy()
    side = int(round(np.sqrt(cls_attn.shape[0])))
    if side * side != cls_attn.shape[0]:
        raise SystemExit(f"non-square patch count: {cls_attn.shape[0]}")
    return cls_attn.reshape(side, side)


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
    side = int(round(np.sqrt(cls.shape[0])))
    return cls.reshape(side, side)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-wrong", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import Dinov2Model

    device = args.device
    print(f"  device: {device}")

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    num_classes = len(idx_to_class)
    class_names = apply_aliases([idx_to_class[str(i)] for i in range(num_classes)])

    print(f"  loading {args.model_id} ...")
    backbone = Dinov2Model.from_pretrained(args.model_id, attn_implementation="eager")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = _load_head(args.run_dir, num_classes).to(device)

    npz = args.run_dir / "test_logits.npz"
    data = np.load(npz, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    preds = logits.argmax(axis=1)
    confidence = (np.exp(logits - logits.max(axis=1, keepdims=True))
                  / np.exp(logits - logits.max(axis=1, keepdims=True))
                  .sum(axis=1, keepdims=True)).max(axis=1)

    splits = json.loads(args.splits.read_text())
    test_samples = splits["test"]
    if len(test_samples) != len(labels):
        raise SystemExit(f"size mismatch: {len(test_samples)} vs {len(labels)}")

    correct_idx = np.where(preds == labels)[0]
    wrong_idx = np.where(preds != labels)[0]
    correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][:args.n_correct]
    wrong_idx = wrong_idx[np.argsort(-confidence[wrong_idx])][:args.n_wrong]

    tf = transforms.Compose([
        transforms.Resize(args.image_size + 32),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(DINOV2_MEAN, DINOV2_STD),
    ])

    rows = max(args.n_correct, args.n_wrong)
    fig, axes = plt.subplots(rows, 6, figsize=(18, rows * 3.0))
    if rows == 1:
        axes = axes[None, :]

    def render(idxs, base_col, kind):
        for r, i in enumerate(idxs):
            sample = test_samples[int(i)]
            with Image.open(sample["path"]).convert("RGB") as im:
                rgb_disp = np.asarray(im.resize((args.image_size, args.image_size)))
            x = tf(Image.open(sample["path"]).convert("RGB")).to(device)

            with torch.no_grad():
                outs = backbone(pixel_values=x.unsqueeze(0),
                                output_attentions=True,
                                interpolate_pos_encoding=True)
            attns_det = [a.detach() for a in outs.attentions]
            rollout = _attentions_to_rollout(attns_det)

            xx = x.unsqueeze(0).clone().detach().requires_grad_(True)
            outs2 = backbone(pixel_values=xx,
                             output_attentions=True,
                             interpolate_pos_encoding=True)
            cls = outs2.last_hidden_state[:, 0]
            logits_t = head(cls)
            target = int(preds[i])
            score = logits_t[0, target]
            grads = torch.autograd.grad(score, outs2.attentions,
                                        retain_graph=False, allow_unused=True)
            grads_filled = [g if g is not None else torch.zeros_like(a)
                            for a, g in zip(outs2.attentions, grads)]
            chefer = _chefer_from_grads([a.detach() for a in outs2.attentions],
                                        grads_filled)

            ov_r = _heat_to_overlay(rgb_disp, rollout)
            ov_c = _heat_to_overlay(rgb_disp, chefer)

            true_n = class_names[int(labels[i])]
            pred_n = class_names[int(preds[i])]
            title = f"true: {true_n}\npred: {pred_n}  p={confidence[i]:.2f}"

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

    fig.suptitle("DINOv2 ViT-B/14 (linear probe) XAI — top-confidence correct (left) vs wrong (right)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "xai_showcase_dinov2_vitb14_linear.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
