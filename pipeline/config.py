from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get(
    "ARCH_DATA_DIR",
    PROJECT_ROOT / "data" / "architectural-styles-dataset",
))
RESULTS_DIR = Path(os.environ.get(
    "ARCH_RESULTS_DIR",
    PROJECT_ROOT / "pipeline" / "results",
))
SPLITS_DIR = RESULTS_DIR / "splits"
RUNS_DIR = RESULTS_DIR / "runs"
SEGMENT_DIR = RESULTS_DIR / "segmentation"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
