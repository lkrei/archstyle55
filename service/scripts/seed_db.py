from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL_SYNC",
                              "postgresql+psycopg://archstyle:archstyle@postgres:5432/archstyle")

ALIAS = {
    "Moscow Luzhkov style architecture": "Late 20th century Moscow architecture",
}

RUNS_BASE = Path(os.environ.get("RUNS_RES_DIR", "/runs_res"))
REPO_BASE = Path(os.environ.get("REPO_DIR", "/repo"))


def _results_root() -> Path:
    override = os.environ.get("RESULTS_DIR")
    if override:
        return Path(override)
    if (RUNS_BASE / "aggregate").is_dir():
        return RUNS_BASE / "aggregate"
    return RUNS_BASE


RESULTS = _results_root()

EMB_FILES = (
    ("test", str(RESULTS / "embeddings/embeddings_test.npz")),
    ("val", str(RESULTS / "embeddings/embeddings_val.npz")),
    ("train", str(RESULTS / "embeddings/embeddings_train.npz")),
)
SUMMARY_CSV = str(RESULTS / "summary_table.csv")
COMPUTE_CSV = str(RESULTS / "compute_cost_table.csv")
IDX_FALLBACKS = (
    str(REPO_BASE / "pipeline/results/splits/idx_to_class.json"),
    str(REPO_BASE / "results/splits/idx_to_class.json"),
    str(RESULTS / "splits/idx_to_class.json"),
)


def _stable_uuid(path: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"archstyle://{path}")


def _sha256(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _ensure_pgvector(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def seed_classes(session: Session) -> int:
    paths = [Path(p) for p in IDX_FALLBACKS]
    src = next((p for p in paths if p.is_file()), None)
    if src is None:
        print("classes: idx_to_class.json not found; skipping")
        return 0
    idx = json.loads(src.read_text(encoding="utf-8"))
    rows = [
        {
            "idx": int(k),
            "name": v,
            "display_name": ALIAS.get(v, v),
        }
        for k, v in idx.items()
    ]
    session.execute(
        text(
            """
            INSERT INTO classes (idx, name, display_name)
            VALUES (:idx, :name, :display_name)
            ON CONFLICT (idx) DO UPDATE SET
              name = EXCLUDED.name,
              display_name = EXCLUDED.display_name
            """
        ),
        rows,
    )
    print(f"classes: upserted {len(rows)}")
    return len(rows)


def seed_models(session: Session) -> int:
    if not Path(SUMMARY_CSV).is_file():
        print("model_meta: no summary_table.csv; skipping")
        return 0
    summary = list(csv.DictReader(open(SUMMARY_CSV, encoding="utf-8")))
    compute = {}
    if Path(COMPUTE_CSV).is_file():
        for row in csv.DictReader(open(COMPUTE_CSV, encoding="utf-8")):
            compute[row.get("model")] = row

    n = 0
    for r in summary:
        name = r.get("model")
        if not name:
            continue
        family = "transformer" if any(s in name for s in ("vit", "swin", "dinov2")) else \
                 ("ensemble" if "ensemble" in name else "cnn")
        c = compute.get(name, {})

        def _f(key, default=None):
            v = r.get(key) or c.get(key)
            try:
                return float(v) if v not in (None, "") else default
            except ValueError:
                return default

        session.execute(
            text(
                """
                INSERT INTO model_meta(name, family, params_m, gflops, accuracy,
                                       macro_f1, bal_acc, inference_ms, image_size, hf_repo)
                VALUES (:name, :family, :params_m, :gflops, :acc, :f1, :bal,
                        :inf, :img, :repo)
                ON CONFLICT (name) DO UPDATE SET
                  accuracy = EXCLUDED.accuracy,
                  macro_f1 = EXCLUDED.macro_f1,
                  bal_acc = EXCLUDED.bal_acc,
                  inference_ms = EXCLUDED.inference_ms,
                  params_m = EXCLUDED.params_m,
                  gflops = EXCLUDED.gflops
                """
            ),
            {
                "name": name,
                "family": family,
                "params_m": _f("params_m", 0.0) or 0.0,
                "gflops": _f("gflops"),
                "acc": _f("accuracy"),
                "f1": _f("macro_f1"),
                "bal": _f("bal_acc"),
                "inf": _f("inference_ms"),
                "img": int(_f("image_size", 224) or 224),
                "repo": os.environ.get("HF_MODEL_REPO", "kkkaredaw/archstyle55-backbones"),
            },
        )
        n += 1
    print(f"model_meta: upserted {n}")
    return n


def seed_embeddings(session: Session, *, max_rows: int | None = None,
                    chunk_size: int = 256) -> int:
    inserted = 0
    seen_sha: set[str] = set()

    def _flush(rows_img, rows_emb):
        if not rows_img:
            return
        session.execute(
            text(
                """
                INSERT INTO images (id, sha256, source, style_label, blob_url)
                VALUES (:id, :sha256, :source, :style_label, :blob_url)
                ON CONFLICT (sha256) DO NOTHING
                """
            ),
            rows_img,
        )
        session.execute(
            text(
                """
                INSERT INTO embeddings (image_id, model, vec)
                VALUES (:image_id, :model, CAST(:vec AS vector))
                ON CONFLICT (image_id, model) DO NOTHING
                """
            ),
            rows_emb,
        )
        session.commit()

    for source, path in EMB_FILES:
        if not Path(path).is_file():
            print(f"embeddings: skip missing {path}")
            continue
        with np.load(path, allow_pickle=True, mmap_mode="r") as d:
            feats = d["features"]
            labels = d["labels"]
            paths = d["paths"]
            class_names = d["class_names"]
            n_rows = len(feats) if max_rows is None else min(len(feats), max_rows)
            print(f"embeddings: ingest {n_rows} from {path}")

            rows_img: list[dict] = []
            rows_emb: list[dict] = []
            for i in range(n_rows):
                p = str(paths[i])
                sha = _sha256(p)
                if sha in seen_sha:
                    continue
                seen_sha.add(sha)
                uid = _stable_uuid(p)
                label_name = str(class_names[int(labels[i])])
                rows_img.append({
                    "id": uid,
                    "sha256": sha,
                    "source": f"split:{source}",
                    "style_label": label_name,
                    "blob_url": p,
                })
                vec_str = "[" + ",".join(f"{float(x):.6f}" for x in feats[i]) + "]"
                rows_emb.append({
                    "image_id": uid,
                    "model": "dinov2_vitb14",
                    "vec": vec_str,
                })
                if len(rows_img) >= chunk_size:
                    _flush(rows_img, rows_emb)
                    inserted += len(rows_img)
                    rows_img.clear()
                    rows_emb.clear()
                    print(f"  inserted {inserted}")
            _flush(rows_img, rows_emb)
            inserted += len(rows_img)
            rows_img.clear()
            rows_emb.clear()
            print(f"  inserted {inserted}")
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap rows per file (debug)")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    _ensure_pgvector(engine)

    with Session(engine) as session:
        seed_classes(session)
        seed_models(session)
        session.commit()
        if not args.skip_embeddings:
            seed_embeddings(session, max_rows=args.max_rows)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
