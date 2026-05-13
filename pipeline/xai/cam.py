
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def _hook_layer(layer):
    activations = {"feat": None, "grad": None}

    def fwd_hook(_, __, output):
        activations["feat"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        activations["grad"] = grad_out[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        yield activations
    finally:
        h1.remove()
        h2.remove()


def grad_cam_pp(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
    image_tensor = image_tensor.requires_grad_(True)

    with _hook_layer(target_layer) as cache:
        logits = model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        feat = cache["feat"][0]
        grad = cache["grad"][0]

    grad2 = grad ** 2
    grad3 = grad ** 3
    sum_a = feat.sum(dim=(1, 2), keepdim=True)
    alpha_denom = 2.0 * grad2 + sum_a * grad3
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = grad2 / alpha_denom
    weights = (alphas * F.relu(grad)).sum(dim=(1, 2))
    cam = (weights[:, None, None] * feat).sum(dim=0)
    cam = F.relu(cam).cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


@torch.no_grad()
def score_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
    batch: int = 32,
) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor

    with _hook_layer(target_layer) as cache:
        logits = model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        feat = cache["feat"][0]

    upsampled = F.interpolate(feat.unsqueeze(0), size=image_tensor.shape[-2:],
                              mode="bilinear", align_corners=False)[0]
    flat = upsampled.view(upsampled.size(0), -1)
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    norm = (flat - mins) / (maxs - mins + 1e-9)
    norm = norm.view_as(upsampled)

    cam_weights = []
    for i in range(0, norm.size(0), batch):
        masks = norm[i:i + batch].unsqueeze(1)
        masked = image_tensor * masks
        scores = F.softmax(model(masked), dim=1)[:, target_class]
        cam_weights.append(scores.cpu())
    cam_weights = torch.cat(cam_weights)
    cam = (cam_weights[:, None, None] * F.relu(feat).cpu()).sum(dim=0)
    cam = F.relu(cam).numpy()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


@torch.no_grad()
def eigen_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor

    with _hook_layer(target_layer) as cache:
        _ = model(image_tensor)
        feat = cache["feat"][0].cpu().numpy()

    c, h, w = feat.shape
    mat = feat.reshape(c, h * w)
    mat_c = mat - mat.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(mat_c, full_matrices=False)
    direction = u[:, 0]
    cam = (direction.reshape(c, 1, 1) * feat).sum(axis=0)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    h, w = image_rgb.shape[:2]
    if heatmap.shape != (h, w):
        from PIL import Image
        heatmap = np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)),
        ) / 255.0
    cmap = cm.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended


def find_default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Эвристика для torchvision-моделей: последний conv-блок до AdaptiveAvgPool."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("no conv layer found")
    return last_conv
