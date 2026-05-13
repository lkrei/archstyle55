"""License-only image expansion (Wikimedia Commons + Openverse).

Only CC0 / Public Domain / CC-BY / CC-BY-SA images are accepted.
License metadata (``license``, ``license_url``, ``attribution``) is
stored in ``images.extra`` for every persisted row.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

import redis
import requests
from sqlalchemy import create_engine, text

from ..core.config import get_settings
from ..core.logging import configure_logging, get_logger
from ..utils.images import load_pil, sha256_hex, to_short_edge

configure_logging()
log = get_logger("license_fetcher")

ACCEPTED_LICENSES = {
    "cc0", "cc-pd", "publicdomain",
    "cc-by", "cc-by-sa", "cc-by-2.0", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-2.0", "cc-by-sa-3.0", "cc-by-sa-4.0",
}


def _redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _engine():
    return create_engine(get_settings().database_url_sync, pool_pre_ping=True)


def _publish(job_id: str, event: dict) -> None:
    r = _redis()
    channel = f"scrape:{job_id}"
    r.publish(channel, json.dumps(event))
    r.lpush(f"scrape:log:{job_id}", json.dumps(event))
    r.ltrim(f"scrape:log:{job_id}", 0, 999)
    r.expire(f"scrape:log:{job_id}", 3600)


def _update_status(job_id: str, *, status: str | None = None, n_done: int | None = None,
                   log_entry: dict | None = None, finished: bool = False) -> None:
    eng = _engine()
    with eng.begin() as conn:
        if status is not None:
            conn.execute(text("UPDATE scrape_jobs SET status=:s WHERE id=:i"),
                         {"s": status, "i": job_id})
        if n_done is not None:
            conn.execute(text("UPDATE scrape_jobs SET n_done=:n WHERE id=:i"),
                         {"n": n_done, "i": job_id})
        if log_entry is not None:
            conn.execute(
                text("UPDATE scrape_jobs SET log = log || CAST(:e AS jsonb) WHERE id=:i"),
                {"e": json.dumps([log_entry]), "i": job_id},
            )
        if finished:
            conn.execute(text("UPDATE scrape_jobs SET finished_at = now() WHERE id=:i"),
                         {"i": job_id})


def _normalize_license(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace(" ", "-")
    if v in ACCEPTED_LICENSES:
        return v
    for prefix in ("cc-by-sa-", "cc-by-"):
        if v.startswith(prefix):
            return v
    if v.startswith("public"):
        return "publicdomain"
    return None


def _wikimedia_search(query: str, n: int) -> Iterator[dict]:
    """Поиск файлов в Wikimedia Commons через action=query|generator=search.

    Возвращает имена страниц File:..., затем подтягивает imageinfo с лицензией.
    """
    settings = get_settings()
    headers = {"User-Agent": settings.scraper_user_agent}
    api = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} architecture",
        "gsrnamespace": "6",
        "gsrlimit": min(50, max(20, n)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 1024,
    }
    try:
        r = requests.get(api, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return

    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata") or {}
        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            continue
        license_short = _normalize_license(
            (meta.get("LicenseShortName") or {}).get("value")
        )
        if license_short is None:
            continue
        yield {
            "src": "wikimedia",
            "page": page.get("title"),
            "image_url": info.get("thumburl") or info.get("url"),
            "license": license_short,
            "license_url": (meta.get("LicenseUrl") or {}).get("value"),
            "attribution": (
                (meta.get("Artist") or {}).get("value")
                or (meta.get("Credit") or {}).get("value")
                or "Wikimedia Commons"
            ),
            "width": info.get("width"),
            "height": info.get("height"),
        }


def _openverse_search(query: str, n: int) -> Iterator[dict]:
    """Openverse: агрегатор CC-лицензий (не требует ключа для базового
    rate-limit'а).
    """
    headers = {"User-Agent": get_settings().scraper_user_agent}
    api = "https://api.openverse.engineering/v1/images/"
    params = {
        "q": f"{query} architecture",
        "license_type": "all-cc",
        "page_size": min(50, max(20, n)),
    }
    try:
        r = requests.get(api, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return

    for it in data.get("results", []):
        license_short = _normalize_license(it.get("license"))
        if license_short is None:
            continue
        yield {
            "src": "openverse",
            "page": it.get("foreign_landing_url"),
            "image_url": it.get("url"),
            "license": license_short,
            "license_url": it.get("license_url"),
            "attribution": (it.get("creator") or "Openverse contributors"),
            "width": it.get("width"),
            "height": it.get("height"),
        }


def _candidates(query: str, n: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for fetch in (_wikimedia_search, _openverse_search):
        for cand in fetch(query, n):
            url = cand.get("image_url")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(cand)
            if len(out) >= n * 2:
                return out
    return out


def _download(url: str, *, timeout: int = 15) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": get_settings().scraper_user_agent
        })
        if r.status_code == 200 and len(r.content) >= 4_000:
            return r.content
    except requests.RequestException:
        return None
    return None


def _exists_sha(eng, sha: str) -> bool:
    with eng.connect() as conn:
        return bool(conn.execute(
            text("SELECT 1 FROM images WHERE sha256=:s"), {"s": sha}
        ).scalar())


def _persist_image(eng, *, sha: str, cand: dict, style: str,
                   width: int, height: int) -> uuid.UUID:
    img_id = uuid.uuid4()
    extra = json.dumps({
        "license": cand["license"],
        "license_url": cand.get("license_url"),
        "attribution": cand.get("attribution"),
        "page": cand.get("page"),
    })
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO images
                  (id, sha256, source, style_label, blob_url, width, height, extra)
                VALUES
                  (:id, :sha, :src, :style, :url, :w, :h, CAST(:extra AS jsonb))
                ON CONFLICT (sha256) DO NOTHING
            """),
            {
                "id": img_id,
                "sha": sha,
                "src": cand["src"],
                "style": style,
                "url": cand["image_url"],
                "w": width,
                "h": height,
                "extra": extra,
            },
        )
    return img_id


def _persist_embedding(eng, *, image_id: uuid.UUID, vec: list[float]) -> None:
    vec_lit = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO embeddings (image_id, model, vec)
                VALUES (:i, 'dinov2_vitb14', CAST(:v AS vector))
                ON CONFLICT (image_id, model) DO NOTHING
            """),
            {"i": image_id, "v": vec_lit},
        )


def run_scrape_job(job_id: str, style: str, query: str, n_target: int) -> dict:
    settings = get_settings()
    n_target = min(n_target, settings.scraper_max_per_job)
    eng = _engine()
    _update_status(job_id, status="running")
    _publish(job_id, {
        "event": "start",
        "n_target": n_target,
        "query": query,
        "style": style,
        "sources": ["wikimedia", "openverse"],
    })

    from ..ml.classify import predict_ensemble
    from ..ml.embed import embed_image

    n_done = 0
    n_skipped = 0
    started = time.time()
    try:
        cands = _candidates(query, max(n_target * 2, n_target + 10))
        _publish(job_id, {"event": "candidates", "count": len(cands)})
        for cand in cands:
            if n_done >= n_target:
                break
            data = _download(cand["image_url"])
            if data is None:
                continue
            sha = sha256_hex(data)
            if _exists_sha(eng, sha):
                n_skipped += 1
                continue
            try:
                img = load_pil(data)
            except ValueError:
                continue
            small = to_short_edge(img, short=384)
            emb = embed_image(small)
            ensemble = predict_ensemble(small, mode="uniform")
            img_id = _persist_image(
                eng, sha=sha, cand=cand, style=style,
                width=img.width, height=img.height,
            )
            _persist_embedding(eng, image_id=img_id, vec=emb.vector)
            n_done += 1
            _update_status(job_id, n_done=n_done)
            _publish(job_id, {
                "event": "ingested",
                "url": cand["image_url"],
                "image_id": str(img_id),
                "license": cand["license"],
                "attribution": cand.get("attribution"),
                "source": cand["src"],
                "top1": ensemble.top1_class,
                "prob": ensemble.top1_prob,
                "n_done": n_done,
                "n_target": n_target,
            })

        elapsed = round(time.time() - started, 2)
        _update_status(
            job_id, status="finished", n_done=n_done, finished=True,
            log_entry={"summary": {"n_done": n_done, "n_skipped": n_skipped,
                                   "elapsed_s": elapsed}},
        )
        _publish(job_id, {"event": "done", "n_done": n_done,
                          "n_skipped": n_skipped, "elapsed_s": elapsed})
        return {"job_id": job_id, "n_done": n_done, "n_skipped": n_skipped,
                "elapsed_s": elapsed}
    except Exception as exc:
        log.error("license_fetch.failed", job_id=job_id, error=str(exc))
        _update_status(job_id, status="failed", finished=True,
                       log_entry={"error": str(exc)})
        _publish(job_id, {"event": "error", "error": str(exc)})
        raise
