from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

SECRET_KEYS = {
    "HF_TOKEN", "WANDB_API_KEY", "PREFECT_API_KEY",
    "DATABASE_URL", "DATABASE_URL_SYNC", "REDIS_URL",
}
VAR_KEYS = {
    "APP_ENV", "LOG_LEVEL", "BACKEND_URL", "MODEL_CACHE_DIR",
    "MODEL_LRU_SLOTS", "INFERENCE_DEVICE", "TORCH_NUM_THREADS",
    "RATE_LIMIT_PREDICT", "ALLOWED_ORIGINS", "SCRAPER_USER_AGENT",
    "SCRAPER_MAX_PER_JOB", "WANDB_PROJECT", "PREFECT_API_URL",
    "HF_MODEL_REPO", "HF_DATASET_REPO", "HF_ORG",
}


def _parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--env-file", default="service/.env")
    args = parser.parse_args()
    assert args.token, "HF_TOKEN required"

    env = _parse_env(Path(args.env_file))
    api = HfApi(token=args.token)

    set_s, set_v = 0, 0
    for k, v in env.items():
        if not v:
            continue
        if k in SECRET_KEYS:
            api.add_space_secret(args.repo, k, v)
            set_s += 1
            print(f"secret  set: {k}")
        elif k in VAR_KEYS:
            api.add_space_variable(args.repo, k, v)
            set_v += 1
            print(f"var     set: {k}={v[:40]}{'...' if len(v) > 40 else ''}")
    print(f"done: {set_s} secrets, {set_v} variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
