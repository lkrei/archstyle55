---
title: ArchStyle55 Backend
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# ArchStyle55 Backend

FastAPI inference + segmentation + XAI + kNN search for 55 architectural styles.

Запросы:
- `POST /predict/single`, `/predict/ensemble`, `/predict/hybrid`, `/predict/zeroshot`, `/predict/all`
- `POST /segment`, `POST /segment/color`
- `POST /xai/cnn`, `POST /xai/transformer`
- `POST /search/similar`, `GET /search/atlas`
- `POST /scrape/start`, `WS /scrape/ws/{job_id}`
- `GET /meta/models`, `/meta/classes`, `/meta/leaderboard`
- `POST /feedback`, `GET /feedback/stats`

Документация: `/docs` (Swagger UI).
