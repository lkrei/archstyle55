from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from ..core.config import get_settings
from ..core.logging import get_logger

log = get_logger(__name__)

LOCAL_RUNS = (
    "_unpacked/run_{run}/{run}/best.pt",
    "{run}/best.pt",
)

HF_FILE_TEMPLATE = "{run}/best.pt"


def resolve_checkpoint(run_name: str) -> Path:
    settings = get_settings()
    base = settings.runs_dir
    if base.is_dir():
        for tmpl in LOCAL_RUNS:
            candidate = base / tmpl.format(run=run_name)
            if candidate.is_file():
                return candidate
    if not settings.hf_token:
        raise FileNotFoundError(
            f"checkpoint for {run_name} not found locally and HF_TOKEN is empty"
        )
    log.info("hf.download", run=run_name)
    path = hf_hub_download(
        repo_id=settings.hf_model_repo,
        filename=HF_FILE_TEMPLATE.format(run=run_name),
        token=settings.hf_token,
        cache_dir=str(settings.model_cache_dir),
    )
    return Path(path)


def maybe_unwrap(state_dict: dict) -> dict:
    for key in ("state", "model_state", "model", "state_dict"):
        if key in state_dict and isinstance(state_dict[key], dict):
            return state_dict[key]
    return state_dict
