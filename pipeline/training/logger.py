
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    val_balanced_acc: float
    lr_head: float
    lr_backbone: float
    grad_norm: float
    epoch_time_s: float
    extra: dict[str, Any] = field(default_factory=dict)


class RunLogger:
    def __init__(self, run_dir: Path, project: str | None = None,
                 run_name: str | None = None, config: dict | None = None,
                 enable_wandb: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.json"
        self._csv_writer = None
        self._csv_file = None
        self._t0 = time.time()
        self.wandb = None

        if config is not None:
            self.config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        if enable_wandb:
            try:
                import wandb
                self.wandb = wandb.init(
                    project=project, name=run_name, dir=str(self.run_dir),
                    config=config or {}, reinit=True,
                )
            except (ImportError, Exception):  # noqa: BLE001
                self.wandb = None

    def log_epoch(self, m: EpochMetrics) -> None:
        if self._csv_writer is None:
            fields = list(asdict(m).keys())
            fields.remove("extra")
            self._csv_file = self.metrics_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._csv_writer.writeheader()

        row = {k: v for k, v in asdict(m).items() if k != "extra"}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.wandb is not None:
            payload = dict(row)
            payload.update({f"extra/{k}": v for k, v in m.extra.items()})
            self.wandb.log(payload, step=m.epoch)

        print(
            f"epoch {m.epoch:3d} | "
            f"train {m.train_loss:.3f} | val {m.val_loss:.3f} | "
            f"acc {m.val_acc:.3f} | f1 {m.val_macro_f1:.3f} | "
            f"lr {m.lr_head:.2e} | t {m.epoch_time_s:.0f}s",
            flush=True,
        )

    def log_image(self, name: str, image, epoch: int | None = None) -> None:
        if self.wandb is None:
            return
        import wandb
        self.wandb.log({name: wandb.Image(image)}, step=epoch)

    def log_artifact(self, path: Path, name: str, artifact_type: str = "result") -> None:
        if self.wandb is None:
            return
        import wandb
        art = wandb.Artifact(name, type=artifact_type)
        art.add_file(str(path))
        self.wandb.log_artifact(art)

    def close(self, summary: dict | None = None) -> None:
        if summary is not None:
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            if self.wandb is not None:
                for k, v in summary.items():
                    self.wandb.summary[k] = v

        if self._csv_file is not None:
            self._csv_file.close()
        if self.wandb is not None:
            self.wandb.finish()
