# Deployment guide

Stack: HF Spaces (backend) + Streamlit Cloud (frontend) + Neon Postgres + Upstash Redis + Prefect Cloud + W&B.

## 0. Tokens

Collect the following secrets in `service/.env`:

| key | где взять |
| --- | --- |
| `HF_TOKEN` | https://huggingface.co/settings/tokens (write) |
| `HF_MODEL_REPO` | e.g. `archstyle55/backbones` |
| `HF_SPACE_REPO` | e.g. `archstyle55/backend` |
| `DATABASE_URL` | Neon connection string `postgresql+asyncpg://...` (with `?sslmode=require`) |
| `DATABASE_URL_SYNC` | same, but `postgresql+psycopg://...` |
| `REDIS_URL` | Upstash Redis URL `rediss://default:<token>@<host>:<port>` |
| `PREFECT_API_URL` | `https://app.prefect.cloud/api/accounts/<acc>/workspaces/<ws>` |
| `PREFECT_API_KEY` | Prefect Cloud -> API Keys |
| `WANDB_API_KEY` | https://wandb.ai/settings |

## 1. Push weights to HF Hub

```bash
cd service
python scripts/push_models_to_hf.py \
  --source <local-runs-dir> \
  --repo-id "$HF_MODEL_REPO" \
  --token "$HF_TOKEN" \
  --include-attributes
```

This creates a repo with every `best.pt` and `attributes.csv`.

## 2. Neon Postgres + pgvector

1. Create a project at https://neon.tech.
2. Open `neondb` -> *Extensions* -> enable `vector`.
3. Store both URLs (`asyncpg`, `psycopg`) in `.env`.

## 3. Upstash Redis

1. https://console.upstash.com -> *Create database*.
2. *Endpoints* -> copy the Redis URL (`rediss://...`) into `REDIS_URL`.

## 4. Migrations and seed

Locally via docker compose:

```bash
make up
make migrate
make seed
docker compose exec backend python -m scripts.bootstrap_classes
docker compose exec backend python -m scripts.train_hybrid
```

The same flow can run against Neon by pointing `DATABASE_URL_SYNC` at
the Neon URL, but a dump-and-restore is faster:

```bash
docker compose exec postgres pg_dump -U archstyle -F c archstyle > pg.dump
pg_restore --no-owner -d "$NEON_PSQL_URL" pg.dump
```

## 5. Deploy backend on HF Spaces

Add `HF_TOKEN` and `HF_SPACE_REPO` to GitHub secrets. The
`deploy-backend.yml` workflow uploads `service/backend/*` to Spaces on
every push to `main`. Or run the script directly:

```bash
python service/scripts/deploy_hf_space.py --repo "$HF_SPACE_REPO"
python service/scripts/set_space_secrets.py --repo "$HF_SPACE_REPO" --env-file service/.env
```

After the Space restarts the backend is served at
`https://<owner>-<repo>.hf.space`.

## 6. Deploy frontend on Streamlit Cloud

1. https://share.streamlit.io -> *New app*.
2. Repo / branch / main file = `service/frontend/Home.py`.
3. *Advanced settings* -> secrets:

```toml
BACKEND_URL = "https://<owner>-<repo>.hf.space"
WANDB_PROJECT = "archstyle-vkr"
PREFECT_API_URL = "https://app.prefect.cloud"
```

## 7. Prefect deployments

```bash
prefect cloud login -k "$PREFECT_API_KEY"
cd service/prefect
python -m flows.scrape_round
python -m flows.reembed_index
python -m flows.recalibrate
python -m flows.drift_report
```

`python deployments.py` registers all flows at once.

## 8. Smoke checklist

```bash
curl https://<backend>/healthz
curl https://<backend>/readyz
curl https://<backend>/meta/models | jq
```

The Streamlit Home page should show a green "backend ok" badge.
