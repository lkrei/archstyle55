from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.ml.labels import class_aliases, display  # noqa: E402


def test_alias_replaces_problem_class():
    aliases = class_aliases()
    raw = "Moscow Luzhkov style architecture"
    assert raw in aliases
    assert aliases[raw] == "Late 20th century Moscow architecture"
    assert display(raw) == "Late 20th century Moscow architecture"


def test_alias_passes_unknown_unchanged():
    assert display("Stalinist architecture") == "Stalinist architecture"
