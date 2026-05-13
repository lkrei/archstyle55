from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..core.config import get_settings

_HERE = Path(__file__).resolve()
_APP_ROOT = _HERE.parents[2]  # service/backend/app/ml/labels.py -> service/backend

CANDIDATES = (
    _APP_ROOT / "results" / "splits" / "idx_to_class.json",
    _APP_ROOT.parent / "results" / "splits" / "idx_to_class.json",
    Path("/runs_res/_local") / "idx_to_class.json",
    Path("/runs_res/aggregate/_local/idx_to_class.json"),
    Path("/runs_res/idx_to_class.json"),
    Path("/app/results/splits/idx_to_class.json"),
    Path("/repo/results/splits/idx_to_class.json"),
    Path("/repo/pipeline/results/splits/idx_to_class.json"),
)


@lru_cache
def load_idx_to_class() -> dict[int, str]:
    settings = get_settings()
    paths = [settings.runs_dir / "splits" / "idx_to_class.json"]
    paths += list(CANDIDATES)
    for p in paths:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return {int(k): v for k, v in data.items()}
    raise FileNotFoundError(f"idx_to_class.json not found in {paths}")


@lru_cache
def class_aliases() -> dict[str, str]:
    return {
        "Moscow Luzhkov style architecture": "Late 20th century Moscow architecture",
    }


@lru_cache
def class_names_raw() -> list[str]:
    idx = load_idx_to_class()
    return [idx[i] for i in range(len(idx))]


@lru_cache
def class_names() -> list[str]:
    aliases = class_aliases()
    return [aliases.get(n, n) for n in class_names_raw()]


def display(name: str) -> str:
    return class_aliases().get(name, name)
