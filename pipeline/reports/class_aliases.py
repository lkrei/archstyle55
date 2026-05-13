
from __future__ import annotations

import json
import os
from pathlib import Path

CLASS_ALIASES: dict[str, str] = {
    # Класс изначально содержал смесь стилей разных эпох, объединённых
    # географически и хронологически (Москва, поздний XX в.).
    "Moscow Luzhkov style architecture": "Late 20th century Moscow architecture",
}


def _load_extra() -> dict[str, str]:
    path = os.environ.get("ARCH_CLASS_ALIASES")
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def all_aliases() -> dict[str, str]:
    merged = dict(CLASS_ALIASES)
    merged.update(_load_extra())
    return merged


def apply_aliases(names):
    aliases = all_aliases()
    if isinstance(names, str):
        return aliases.get(names, names)
    if isinstance(names, dict):
        return {k: aliases.get(v, v) for k, v in names.items()}
    return [aliases.get(n, n) for n in names]


def remap_report_keys(report: dict) -> dict:
    aliases = all_aliases()
    out = {}
    for k, v in report.items():
        new_k = aliases.get(k, k)
        out[new_k] = v
    return out
