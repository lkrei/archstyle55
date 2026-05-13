
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.models as tvm
from torch import nn

from ..models.factory import DEFAULT_HPARAMS, build_model


def _build_no_weights(name: str, num_classes: int) -> nn.Module:

    if name == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "efficientnet_b2":
        m = tvm.efficientnet_b2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "efficientnet_b3":
        m = tvm.efficientnet_b3(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "convnext_small":
        m = tvm.convnext_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "vit_b16":
        m = tvm.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    if name == "swin_v2_t":
        m = tvm.swin_v2_t(weights=None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: torch.nn.Module, image_size: int) -> float | None:
    x = torch.randn(1, 3, image_size, image_size)
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flops = FlopCountAnalysis(model, x).total()
        return float(flops) / 1e9
    except (ImportError, Exception):  # noqa: BLE001
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(model, (3, image_size, image_size),
                                               as_strings=False, print_per_layer_stat=False,
                                               verbose=False)
            return float(macs) * 2 / 1e9
        except (ImportError, Exception):  # noqa: BLE001
            return None


def inference_ms(model: torch.nn.Module, image_size: int, batch_size: int = 1,
                 warmup: int = 5, repeats: int = 30, device: str = "cpu") -> float:
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return float(elapsed / repeats * 1000.0 / batch_size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_HPARAMS))
    parser.add_argument("--num-classes", type=int, default=55)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("compute_cost.json"))
    parser.add_argument("--no-pretrained", action="store_true",
                        help="строить архитектуру без скачивания весов (быстрый CPU bench)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in args.models:
        spec = DEFAULT_HPARAMS[name]
        try:
            if args.no_pretrained:
                model = _build_no_weights(name, num_classes=args.num_classes)
            else:
                model = build_model(name, num_classes=args.num_classes)
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": name, "error": str(exc)})
            continue
        try:
            params = count_parameters(model)
            gflops = count_flops(model, spec.image_size)
            ms = inference_ms(model, spec.image_size, batch_size=1, device=device)
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": name, "image_size": spec.image_size, "error": str(exc)})
            continue
        rows.append({
            "model": name,
            "image_size": spec.image_size,
            "params_m": round(params / 1e6, 2),
            "gflops": round(gflops, 3) if gflops is not None else None,
            "inference_ms": round(ms, 2),
        })
        print(rows[-1], flush=True)

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
