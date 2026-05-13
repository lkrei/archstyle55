# Live deployment

Reference layout used during development. Adapt names, regions and
secrets to your own accounts.

## Components

| Component | URL |
| --- | --- |
| HF Hub model repo | `<user>/archstyle55-backbones` |
| HF Space (backend) | `<user>/archstyle55-backend` |
| Public backend URL | `https://<user>-archstyle55-backend.hf.space` |
| Neon Postgres + pgvector | `eu-west-2` |
| Upstash Redis | `eu-central-1` |
| Streamlit Community Cloud | requires a public GitHub fork of `service/frontend/` |
| Prefect Cloud | scheduled flows under `service/prefect/` |

Tokens, DB URLs and Redis URLs go to `service/.env` (see `.env.example`).

## Push backend to HF Space

```bash
python service/scripts/deploy_hf_space.py --repo "$HF_SPACE_REPO"
```

Re-run on every change in `service/backend/app/**` or `pipeline/**`.

## Sync Space secrets / variables

```bash
python service/scripts/set_space_secrets.py \
  --repo "$HF_SPACE_REPO" \
  --env-file service/.env
```

`HF_TOKEN`, `WANDB_API_KEY`, `PREFECT_API_KEY`, `DATABASE_URL*`,
`REDIS_URL` go in as secrets; everything else as variables.

## Migrate Neon and seed

```bash
export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')
export PGSSLROOTCERT=$SSL_CERT_FILE
export DATABASE_URL_SYNC='<neon postgres URL>'

cd service/backend && alembic upgrade head
cd ..
PYTHONPATH=backend python -u scripts/seed_db.py
PYTHONPATH=backend python -u scripts/bootstrap_classes.py
```

## Streamlit Community Cloud

Streamlit Cloud only deploys from a public GitHub repo. Either keep
this repo public, or fork `service/frontend/` into a separate public
repo, then on `streamlit.io/cloud`:

- Main file: `Home.py`
- Python: 3.11
- Secrets:

```toml
BACKEND_URL = "https://<user>-archstyle55-backend.hf.space"
```

## Prefect Cloud

```bash
prefect cloud login -k "$PREFECT_API_KEY"
prefect work-pool create --type process default-pool
cd service/backend
python flows/deployments.py
```

Registers `daily_licensed_round`, `nightly_reembed`,
`weekly_recalibrate`, `drift_report`.

## Local fallback

`make up` brings Postgres + Redis + Backend + Streamlit through
docker-compose. Use local URLs in `service/.env`.
